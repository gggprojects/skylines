
#include <fstream>
#include <iostream>

#include <celero/Celero.h>
#include <csv.h>
#include <boost/program_options.hpp>

#include "queries/weighted.hpp"

struct DataSize {
    DataSize() {}
    DataSize(size_t input_p_size, size_t input_q_size) : input_p_size_(input_p_size), input_q_size_(input_q_size) {
    }
    size_t input_p_size_;
    size_t input_q_size_;
};

std::string GetFileName(const DataSize &ds) {
    return std::move(std::to_string(ds.input_p_size_) + "x" + std::to_string(ds.input_q_size_) + ".bin");
}

sl::queries::WeightedQuery wq(sl::error::ThreadsErrors::Instanciate());
std::map<int64_t, DataSize> experiment_value_filename_map;

void GenerateFile(size_t input_p_size, size_t input_q_size) {
    DataSize ds(input_p_size, input_q_size);
    wq.InitRandom(ds.input_p_size_, ds.input_q_size_);
    uint64_t current_size = experiment_value_filename_map.size();
    experiment_value_filename_map.insert(std::pair<int64_t, DataSize>(experiment_value_filename_map.size(), ds));
    wq.ToFile(GetFileName(experiment_value_filename_map[current_size]));
}

void GenerateFiles() {
    //GenerateFile(500, 10);
    //GenerateFile(1000, 10);
    //GenerateFile(5000, 10);
    //GenerateFile(10000, 10);
    //GenerateFile(100000, 10);
    GenerateFile(1, 1);
    GenerateFile(2, 1);
    GenerateFile(3, 1);
    GenerateFile(4, 1);
    GenerateFile(5, 1);
}

void TransformCSV(const std::string &filename) {
    std::string output;
    io::CSVReader<17> in(filename);
    in.read_header(io::ignore_extra_column, "Group", "Experiment", "Problem Space", "Samples", "Iterations", "Failure", "Baseline", "us/Iteration", "Iterations/sec", "Min (us)", "Mean (us)", "Max (us)", "Variance", "Standard Deviation", "Skewness", "Kurtosis", "Z Score");
    output = "PxQ,Experiment,Improvement,Min Running Time(s),Max Running Time(s),Mean Running Time(s),Variance,Standard Deviation,Skewness,Kurtosis,Z Score\n";
    std::string group, experiment;
    double problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_core;
    while (in.read_row(group, experiment, problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_core)) {
        std::string line;

        line += std::to_string(experiment_value_filename_map[problem_space].input_p_size_) + "x" + std::to_string(experiment_value_filename_map[problem_space].input_q_size_) + ",";
        line += experiment + ",";
        line += std::to_string(baseline) + ",";
        line += std::to_string(min_us) + ",";
        line += std::to_string(max_us) + ",";
        line += std::to_string(mean_us) + ",";
        line += std::to_string(variance) + ",";
        line += std::to_string(standard_deviation) + ",";
        line += std::to_string(skewness) + ",";
        line += std::to_string(kurtosis) + ",";
        line += std::to_string(z_core) + ",";
        line += "\n";
        output += line;
    }

    {
        std::ofstream output_file(filename + ".transformed.csv");
        output_file << output;
        output_file.close();
    }
}

int main(int argc, char** argv) {
    GenerateFiles();
    celero::Run(argc, argv);

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("t", boost::program_options::value<std::string>(), "Run and create CSV");

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc)
            .style(boost::program_options::command_line_style::default_style | boost::program_options::command_line_style::allow_long_disguise)
            .run(), vm);
        boost::program_options::notify(vm);
    } catch (const std::exception &e) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("t")) {
        std::string filename = vm["t"].as<std::string>();
        TransformCSV(filename);
    }

    return 0;
}

class InitFromBinaryFileFixture : public celero::TestFixture {
public:

    InitFromBinaryFileFixture() : wq(sl::error::ThreadsErrors::Instanciate()) {
    }

    virtual std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override {
        std::vector<std::pair<int64_t, uint64_t>> problemSpace;

        std::vector<int> iterations { 80, 40, 10, 2, 1};
        std::vector<int>::iterator it = iterations.begin();
        for (const std::pair<int64_t, DataSize> &kvp : experiment_value_filename_map) {
            problemSpace.push_back(std::make_pair(kvp.first, *it));
            ++it;
        }
        return problemSpace;
    }

    void setUp(int64_t experimentValue) override {
        wq.FromFile(GetFileName(experiment_value_filename_map[experimentValue]));
    }

    void tearDown() override {
        wq.Clear();
    }

protected:
    sl::queries::WeightedQuery wq;
};

BASELINE_F(SkylineComputation, Baseline, InitFromBinaryFileFixture, 5, 10) {
    wq.RunSingleThreadBruteForce();
}

BENCHMARK_F(SkylineComputation, SingleThreadBruteForceDiscarting, InitFromBinaryFileFixture, 5, 10) {
    wq.RunSingleThreadBruteForceDiscarting();
}

BENCHMARK_F(SkylineComputation, SingleThreadSorting, InitFromBinaryFileFixture, 5, 10) {
    wq.RunSingleThreadSorting();
}

BENCHMARK_F(SkylineComputation, MultiThreadBruteForce, InitFromBinaryFileFixture, 5, 10) {
    wq.RunMultiThreadBruteForce();
}
