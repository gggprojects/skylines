
#include <fstream>
#include <iostream>

#include <celero/Celero.h>

#include "queries/weighted.hpp"


sl::queries::WeightedQuery wq(sl::error::ThreadsErrors::Instanciate());
std::map<int64_t, std::string> experiment_value_filename_map;

void GenerateFiles() {
    experiment_value_filename_map.insert(std::pair<int64_t, std::string>(0, "10x1000.bin"));
    experiment_value_filename_map.insert(std::pair<int64_t, std::string>(1, "10x10000.bin"));
    experiment_value_filename_map.insert(std::pair<int64_t, std::string>(2, "1000x10.bin"));
    experiment_value_filename_map.insert(std::pair<int64_t, std::string>(3, "10000x10.bin"));

    wq.InitRandom(10, 1000);
    wq.ToFile(experiment_value_filename_map[0]);

    wq.InitRandom(10, 10000);
    wq.ToFile(experiment_value_filename_map[1]);

    wq.InitRandom(1000, 100);
    wq.ToFile(experiment_value_filename_map[2]);

    wq.InitRandom(10000, 10);
    wq.ToFile(experiment_value_filename_map[3]);
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
        problemSpace.push_back(std::make_pair(0, 1000));
        problemSpace.push_back(std::make_pair(1, 1000));
        problemSpace.push_back(std::make_pair(2, 10));
        problemSpace.push_back(std::make_pair(3, 4));
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
