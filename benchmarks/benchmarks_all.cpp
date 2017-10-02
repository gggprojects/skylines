
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


struct Experiment {
public:
    Experiment(size_t input_p_size, size_t input_q_size, std::string experiment_name) :
        input_p_size_(input_p_size), input_q_size_(input_q_size), experiment_name_(experiment_name) {
    }

    size_t input_p_size_;
    size_t input_q_size_;
    std::string experiment_name_;
    double baseline_;
    double max_usec_;
    double min_usec_;
    double mean_usec_;
    double variance_;
    double standard_deviation_;
    double skewness_;
    double kurtosis_;
    double z_score_;
    std::vector<size_t> output_size_;

    std::string GetSizeString() const {
        return std::move(std::to_string(input_p_size_) + "x" + std::to_string(input_q_size_));
    }

    std::string GetFileName() const {
        return std::move(GetSizeString() + ".bin");
    }
};

sl::queries::WeightedQuery wq;
std::multimap<int64_t, Experiment> experiments;

void GenerateFile(size_t input_p_size, size_t input_q_size, uint64_t problem_space) {
    wq.InitRandom(input_p_size, input_q_size);

    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadBruteForce")));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadBruteForceDiscarting")));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadSorting")));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "MultiThreadBruteForce")));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "GPUBruteForce")));

    std::string filename = experiments.equal_range(problem_space).first->second.GetFileName();
    wq.ToFile(filename);
}

void GenerateFiles() {
    GenerateFile(2000, 100, 0);
    GenerateFile(5000, 100, 1);
    GenerateFile(10000, 100, 2);
    GenerateFile(20000, 100, 3);
    GenerateFile(50000, 100, 4);
    GenerateFile(75000, 100, 5);
    GenerateFile(100000, 100, 6);

    //GenerateFile(1000, 10, 0);
    //GenerateFile(2000, 10, 1);
    //GenerateFile(3000, 10, 2);
    //GenerateFile(4, 10, 3);
    //GenerateFile(5, 10, 4);
    //GenerateFile(6, 10, 5);
    //GenerateFile(7, 10, 6);
}

std::string GetHeader() {
    std::string header = "Input size";

    using map_it = std::multimap<int64_t, Experiment>::iterator;
    std::pair<map_it, map_it> iterators = experiments.equal_range(0);
    while (iterators.first != iterators.second) {
        header += ',' + iterators.first->second.experiment_name_;
        iterators.first++;
    }
    header += '\n';
    return std::move(header);
}

void WriteFile(const std::string &filename,
    std::function<double(const Experiment&)> f) {

    std::string output;
    output += GetHeader();

    uint64_t value = 0;
    using map_it = std::multimap<int64_t, Experiment>::iterator;
    std::pair<map_it, map_it> iterators = experiments.equal_range(value);
    while (iterators.first != experiments.end()) {
        output += iterators.first->second.GetSizeString();
        while (iterators.first != iterators.second) {
            output += ',' + std::to_string(f(iterators.first->second));
            iterators.first++;
        }
        output += '\n';
        iterators = experiments.equal_range(++value);
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

    io::CSVReader<17> in(filename);
    in.read_header(io::ignore_extra_column, "Group", "Experiment", "Problem Space", "Samples", "Iterations", "Failure", "Baseline", "us/Iteration", "Iterations/sec", "Min (us)", "Mean (us)", "Max (us)", "Variance", "Standard Deviation", "Skewness", "Kurtosis", "Z Score");
    std::string group, experiment;
    double problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_score;
    while (in.read_row(group, experiment, problem_space, samples, iterations, failure, baseline, us_per_iteration, iterations_sec, min_us, mean_us, max_us, variance, standard_deviation, skewness, kurtosis, z_score)) {
        std::multimap<int64_t, Experiment>::iterator it = experiments.equal_range(static_cast<uint64_t>(problem_space)).first;
        while (it->second.experiment_name_ != experiment) {
            it++;
        }

        it->second.baseline_ = baseline;
        it->second.max_usec_ = max_us;
        it->second.min_usec_ = min_us;
        it->second.mean_usec_ = mean_us;
        it->second.variance_ = variance;
        it->second.standard_deviation_ = standard_deviation;
        it->second.skewness_ = skewness;
        it->second.kurtosis_ = kurtosis;
        it->second.z_score_ = z_score;
    }

    WriteFile(filename + ".transformed-improvement.csv", [](const Experiment &e) -> double { return e.baseline_; });
    WriteFile(filename + ".transformed-running-time-min.csv", [](const Experiment &e) -> double { return e.min_usec_ / 1000000.0; });
    WriteFile(filename + ".transformed-variance.csv", [](const Experiment &e) -> double { return e.variance_; });
    WriteFile(filename + ".transformed-ouput_size.csv", [](const Experiment &e) -> size_t { return e.output_size_[0]; });
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

        int64_t value = 0;
        std::vector<int> iterations{ 100, 80, 50, 30, 10, 5, 1 };
        using map_it = std::multimap<int64_t, Experiment>::iterator;
        std::pair<map_it, map_it> iterators = experiments.equal_range(value);
        do {
            problemSpace.push_back(std::make_pair(iterators.first->first, iterations[value]));
            value++;
            iterators = experiments.equal_range(value);
        } while ((iterators.first != experiments.end()));

        return problemSpace;
    }

    void setUp(int64_t experimentValue) override {
        std::string filename = experiments.equal_range(experimentValue).first->second.GetFileName();
        wq.FromFile(filename);
    }

    void tearDown() override {
        wq.Clear();
    }

protected:
    sl::queries::WeightedQuery wq;
};

void AddOutputSize(size_t input_q_size, std::string experiment_name, size_t output_size) {
    for (std::multimap<int64_t, Experiment>::iterator it = experiments.begin(); it != experiments.end(); it++) {
        if (it->second.input_p_size_ == input_q_size && it->second.experiment_name_ == experiment_name) {
            it->second.output_size_.push_back(output_size);
            return;
        }
    }
}


sl::queries::algorithms::DistanceType distance_type = sl::queries::algorithms::DistanceType::Furthest;

BASELINE_F(SkylineComputation, SingleThreadBruteForce, InitFromBinaryFileFixture, 5, 10) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type);
    AddOutputSize(wq.GetInputP().GetPoints().size(), "SingleThreadBruteForce", wq.GetOuput().GetPoints().size());
}

BENCHMARK_F(SkylineComputation, SingleThreadBruteForceDiscarting, InitFromBinaryFileFixture, 5, 10) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING, distance_type);
    AddOutputSize(wq.GetInputP().GetPoints().size(), "SingleThreadBruteForceDiscarting", wq.GetOuput().GetPoints().size());
}

BENCHMARK_F(SkylineComputation, SingleThreadSorting, InitFromBinaryFileFixture, 5, 10) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type);
    AddOutputSize(wq.GetInputP().GetPoints().size(), "SingleThreadSorting", wq.GetOuput().GetPoints().size());
}

BENCHMARK_F(SkylineComputation, MultiThreadBruteForce, InitFromBinaryFileFixture, 5, 10) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, distance_type);
    AddOutputSize(wq.GetInputP().GetPoints().size(), "MultiThreadBruteForce", wq.GetOuput().GetPoints().size());
}

//BENCHMARK_F(SkylineComputation, MultiThreadSorting, InitFromBinaryFileFixture, 5, 10) {
//    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING);
//}

BENCHMARK_F(SkylineComputation, GPUBruteForce, InitFromBinaryFileFixture, 5, 10) {
    wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type);
    AddOutputSize(wq.GetInputP().GetPoints().size(), "GPUBruteForce", wq.GetOuput().GetPoints().size());
}
