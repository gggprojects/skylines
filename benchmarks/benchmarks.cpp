
#include <fstream>

#include <celero/Celero.h>

#include "queries/weighted.hpp"

CELERO_MAIN

class InitRandomFixture : public celero::TestFixture {
public:

    InitRandomFixture() : wq(sl::error::ThreadsErrors::Instanciate()) {
        wq.InitRandom(10, 10);
    }

    ~InitRandomFixture() {
        wq.Clear();
    }

    std::string ReadAllFile(const std::string &filename) {
        std::ifstream t(filename);
        std::string json_str((std::istreambuf_iterator<char>(t)),
            std::istreambuf_iterator<char>());
        return std::move(json_str);
    }

    void setUp(int64_t experimentValue) override {
        //std::string json_str = ReadAllFile("first_dominated.json");
        //wq.FromJson(json_str);
    }

    void tearDown() override {
        //wq.Clear();
    }

protected:
    sl::queries::WeightedQuery wq;
};

/// In reality, all of the "Complex" cases take the same amount of time to run.
/// The difference in the results is a product of measurement error.
///
/// Interestingly, taking the sin of a constant number here resulted in a 
/// great deal of optimization in clang and gcc.
BASELINE_F(DemoSimple, Baseline, InitRandomFixture, 10, 100) {
    wq.RunSingleThreadSorting();
}

///
/// Run a test consisting of 1 sample of 710000 operations per measurement.
/// There are not enough samples here to likely get a meaningful result.
///
BENCHMARK_F(DemoSimple, Complex1, InitRandomFixture, 10, 100) {
    wq.RunSingleThreadSorting();
}

