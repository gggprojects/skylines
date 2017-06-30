
#include <fstream>
#include <iostream>

#include <celero/Celero.h>

#include "queries/weighted.hpp"


sl::queries::WeightedQuery wq(sl::error::ThreadsErrors::Instanciate());
std::map<int64_t, std::string> experiment_value_filename_map;

void GenerateFile(size_t input_p_size, size_t input_q_size) {
    wq.InitRandom(input_p_size, input_q_size);
    std::string filename = std::to_string(input_p_size) + "x" + std::to_string(input_q_size) + ".bin";
    wq.ToFile(experiment_value_filename_map.insert(std::pair<int64_t, std::string>(experiment_value_filename_map.size(), filename)).first->second);
}

void GenerateFiles() {
    GenerateFile(500, 10);
    GenerateFile(1000, 10);
    GenerateFile(10000, 10);
}

int main(int argc, char** argv) {
    GenerateFiles();
    celero::Run(argc, argv);
    return 0;
}

class InitFromBinaryFileFixture : public celero::TestFixture {
public:

    InitFromBinaryFileFixture() : wq(sl::error::ThreadsErrors::Instanciate()) {
    }

    virtual std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override {
        std::vector<std::pair<int64_t, uint64_t>> problemSpace;

        for (const std::pair<int64_t, std::string> &kvp : experiment_value_filename_map) {
            problemSpace.push_back(std::make_pair(kvp.first, 5));
        }
        return problemSpace;
    }

    void setUp(int64_t experimentValue) override {
        wq.FromFile(experiment_value_filename_map[experimentValue]);
    }

    void tearDown() override {
        wq.Clear();
    }

protected:
    sl::queries::WeightedQuery wq;
};

BASELINE_F(SkylineComputation, Baseline, InitFromBinaryFileFixture, 10, 10) {
    wq.RunSingleThreadBruteForce();
}

BENCHMARK_F(SkylineComputation, SingleThreadBruteForceDiscarting, InitFromBinaryFileFixture, 10, 10) {
    wq.RunSingleThreadBruteForceDiscarting();
}

BENCHMARK_F(SkylineComputation, SingleThreadSorting, InitFromBinaryFileFixture, 10, 10) {
    wq.RunSingleThreadSorting();
}

BENCHMARK_F(SkylineComputation, MultiThreadBruteForce, InitFromBinaryFileFixture, 10, 10) {
    wq.RunMultiThreadBruteForce();
}
