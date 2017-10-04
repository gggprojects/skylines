
#include <fstream>
#include <iostream>
#include <functional>
#include <map>

#pragma warning(push, 0)
#include <celero/Celero.h>
#include <csv.h>
#include <boost/program_options.hpp>
#pragma warning(pop)

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

sl::queries::WeightedQuery wq;
std::map<int64_t, DataSize> experiment_value_filename_map;

void GenerateFile(size_t input_p_size, size_t input_q_size) {
    DataSize ds(input_p_size, input_q_size);
    wq.InitRandom(ds.input_p_size_, ds.input_q_size_);
    uint64_t current_size = experiment_value_filename_map.size();
    experiment_value_filename_map.insert(std::pair<int64_t, DataSize>(experiment_value_filename_map.size(), ds));
    wq.ToFile(GetFileName(experiment_value_filename_map[current_size]));
}

void GenerateFiles() {
    GenerateFile(100000, 10);
    GenerateFile(250000, 10);
    GenerateFile(500000, 10);
    GenerateFile(750000, 10);
    GenerateFile(1000000, 10);
    GenerateFile(2000000, 10);
    GenerateFile(5000000, 10);
    GenerateFile(10000000, 10);
}

struct Experiment {
    std::string input_size_;
    std::string experiment_;
    double baseline_;
    double max_usec_;
    double min_usec_;
    double mean_usec_;
    double variance_;
    double standard_deviation_;
    double skewness_;
    double kurtosis_;
    double z_score_;
};

std::string GetHeader(const std::vector<Experiment> &experiments) {
    std::string header = "Input size,Baseline (Brute force)";
    int pos = 1;
    while (pos < experiments.size() && experiments[pos].experiment_ != "Baseline") {
        header += ',' + experiments[pos].experiment_;
        pos++;
    }
    header += '\n';
    return std::move(header);
}

void WriteFile(const std::vector<Experiment> &experiments, const std::string &filename, std::function<double(const Experiment&)> f) {
    std::string output;
    output += GetHeader(experiments);

    int pos = 0;
    while (pos < experiments.size()) {
        std::string current_input_size = experiments[pos].input_size_;
        output += current_input_size;
        while (pos < experiments.size() && experiments[pos].input_size_ == current_input_size) {
            output += ',' + std::to_string(f(experiments[pos]));
            pos++;
        }
        output += '\n';
    }

    std::ofstream output_file(filename);
    output_file << output;
    output_file.close();
}

void replaceAll(std::string *str, const std::string &from, const std::string &to) {
    if (from.empty())
        return;
    size_t start_pos = 0;
    while ((start_pos = str->find(from, start_pos)) != std::string::npos) {
        str->replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

void PreProcessFile(const std::string &filename) {
    std::ifstream t(filename);
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();

    replaceAll(&str, ",-nan(ind),", ",0,");
    replaceAll(&str, ",nan(ind),", ",0,");
    replaceAll(&str, ",inf,", ",0,");

    std::ofstream t2(filename);
    t2 << str;
    t2.close();
}

void TransformCSV(const std::string &filename) {

    PreProcessFile(filename);

    std::vector<Experiment> experiments;
    io::CSVReader<17> in(filename);
    in.read_header(io::ignore_extra_column, "Group", "Experiment", "Problem Space", "Samples", "Iterations", "Failure", "Baseline", "us/Iteration", "Iterations/sec", "Min (us)", "Mean (us)", "Max (us)", "Variance", "Standard Deviation", "Skewness", "Kurtosis", "Z Score");
    std::string group, experiment;
    double problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_score;
    while (in.read_row(group, experiment, problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_score)) {
        Experiment e;
        e.input_size_ = std::to_string(experiment_value_filename_map[static_cast<uint64_t>(problem_space)].input_p_size_) + "x" + std::to_string(experiment_value_filename_map[static_cast<uint64_t>(problem_space)].input_q_size_);
        e.experiment_ = experiment;
        e.baseline_ = baseline;
        e.max_usec_ = max_us;
        e.min_usec_ = min_us;
        e.mean_usec_ = mean_us;
        e.variance_ = variance;
        e.standard_deviation_ = standard_deviation;
        e.skewness_ = skewness;
        e.kurtosis_ = kurtosis;
        e.z_score_ = z_score;
        experiments.push_back(e);
    }

    //sort by input size
    std::stable_sort(experiments.begin(), experiments.end(), [](const Experiment &a, const Experiment &b) -> bool {
        int a_p_size = std::stoi(a.input_size_.substr(0, a.input_size_.find('x')));
        int b_p_size = std::stoi(b.input_size_.substr(0, b.input_size_.find('x')));
        return a_p_size < b_p_size;
    });

    WriteFile(experiments, filename + ".transformed-improvement.csv", [](const Experiment &e) -> double { return e.baseline_; });
    WriteFile(experiments, filename + ".transformed-running-time-min.csv", [](const Experiment &e) -> double { return e.min_usec_ / 1000000.0; });
    //WriteFile(experiments, filename + ".transformed-running-time-max.csv", [](const Experiment &e) -> double { return e.min_usec_ / 1000000.0; });
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
        (void)e;
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

    InitFromBinaryFileFixture() {
    }

    virtual std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override {
        std::vector<std::pair<int64_t, uint64_t>> problemSpace;

        std::vector<int> iterations{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
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

sl::queries::algorithms::DistanceType distance_type = sl::queries::algorithms::DistanceType::Neartest;

BASELINE_F(SkylineComputation, SingleThreadSorting, InitFromBinaryFileFixture, 1, 1) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type);
}

BENCHMARK_F(SkylineComputation, GPUBruteForce, InitFromBinaryFileFixture, 1, 1) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type);
}
