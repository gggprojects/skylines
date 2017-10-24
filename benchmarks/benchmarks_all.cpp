
#include <fstream>
#include <iostream>
#include <functional>
#include <map>

#pragma warning(push, 0)
#include <celero/Celero.h>
#include <csv.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#pragma warning(pop)

#include "queries/weighted.hpp"


struct Experiment {
public:
    Experiment(size_t input_p_size, size_t input_q_size, std::string experiment_name, sl::queries::algorithms::DistanceType distance_type) :
        input_p_size_(input_p_size), input_q_size_(input_q_size), experiment_name_(experiment_name), distance_type_(distance_type) {
    }

    size_t input_p_size_;
    size_t input_q_size_;
    std::string experiment_name_;
    sl::queries::algorithms::DistanceType distance_type_;
    double baseline_;
    double max_usec_;
    double min_usec_;
    double mean_usec_;
    double variance_;
    double standard_deviation_;
    double skewness_;
    double kurtosis_;
    double z_score_;
    std::vector<sl::queries::data::Statistics> statistics_;

    std::string GetSizeString() const {
        return std::move(std::to_string(input_p_size_) + "x" + std::to_string(input_q_size_));
    }

    std::string GetFileName() const {
        return std::move(GetSizeString() + ".bin");
    }
};

sl::queries::WeightedQuery wq;
std::multimap<int64_t, Experiment> experiments;
sl::queries::algorithms::DistanceType distance_type;

void GenerateFile(size_t input_p_size, size_t input_q_size, uint64_t problem_space, bool create_files) {
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadBruteForce", distance_type)));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadBruteForceDiscarting", distance_type)));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "SingleThreadSorting", distance_type)));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "MultiThreadBruteForce", distance_type)));
    //experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "MultiThreadSorting", distance_type)));
    experiments.insert(std::make_pair(problem_space, Experiment(input_p_size, input_q_size, "GPUBruteForce", distance_type)));

    if (create_files) {
        sl::queries::data::UniformRealRandomGenerator rrg_x(0., 1.);
        sl::queries::data::UniformRealRandomGenerator rrg_y(0., 1.);
        sl::queries::data::UniformIntRandomGenerator irg(1, 10);

        wq.InitRandom(input_p_size, input_q_size, rrg_x, rrg_y, irg);

        std::string filename = experiments.equal_range(problem_space).first->second.GetFileName();
        wq.ToFile(filename);
    }
}

void GenerateFiles(bool create_files) {
    GenerateFile(2000, 100, 0, create_files);
    GenerateFile(5000, 100, 1, create_files);
    GenerateFile(10000, 100, 2, create_files);
    GenerateFile(20000, 100, 3, create_files);
    GenerateFile(50000, 100, 4, create_files);
    GenerateFile(75000, 100, 5, create_files);
    GenerateFile(100000, 100, 6, create_files);

    //GenerateFile(10, 10, 0, create_files);
    //GenerateFile(20, 10, 1, create_files);
    //GenerateFile(30, 10, 2, create_files);
    //GenerateFile(40, 10, 3, create_files);
    //GenerateFile(50, 10, 4, create_files);
    //GenerateFile(70, 10, 5, create_files);
    //GenerateFile(80, 10, 6, create_files);
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

void TransformCSV(const std::string &filename, const std::string &destination_filename) {

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

    WriteFile(destination_filename + "-improvement.csv", [](const Experiment &e) -> double { return e.baseline_; });
    WriteFile(destination_filename + "-running-time-min.csv", [](const Experiment &e) -> double { return e.min_usec_ / 1000000.0; });
    WriteFile(destination_filename + "-variance.csv", [](const Experiment &e) -> double { return e.variance_; });
    WriteFile(destination_filename + "-ouput_size.csv", [](const Experiment &e) -> size_t { return e.statistics_[0].output_size_; });
    WriteFile(destination_filename + "-num_comparisons.csv", [](const Experiment &e) -> size_t { return e.statistics_[0].num_comparisions_; });

    boost::filesystem::path p(filename);
    boost::filesystem::remove(p);
}

int main(int argc, char** argv) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("t", boost::program_options::value<std::string>(), "Run and create CSV")
        ("d", boost::program_options::value<int>(), "Distance type. 1 nearest, 0 furthest");

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

    std::string filename;
    if (vm.count("t")) {
        filename = vm["t"].as<std::string>();
    } else {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("d")) {
        distance_type = (vm["d"].as<int>() == 1 ? sl::queries::algorithms::DistanceType::Neartest : sl::queries::algorithms::DistanceType::Furthest);
    } else {
        std::cout << desc << "\n";
        return 1;
    }

    GenerateFiles(distance_type == sl::queries::algorithms::DistanceType::Neartest);

    char **argv_copy = new char*[3];
    argv_copy[0] = argv[0];
    argv_copy[1] = argv[1];
    argv_copy[2] = argv[2];
    celero::Run(3, argv_copy);
    TransformCSV(filename, filename + (distance_type == sl::queries::algorithms::DistanceType::Neartest ? "-nearest" : "-furthest"));

    return 0;
}

class InitFromBinaryFileFixture : public celero::TestFixture {
public:

    InitFromBinaryFileFixture() {
    }

    virtual std::vector<std::pair<int64_t, uint64_t>> getExperimentValues() const override {
        std::vector<std::pair<int64_t, uint64_t>> problemSpace;

        int64_t value = 0;
        std::vector<int> iterations{ 100, 80, 10, 5, 3, 2, 1 };
        //std::vector<int> iterations{ 1, 1, 1, 1, 1, 1, 1 };
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

void AddStatistics(size_t input_q_size, std::string experiment_name, const sl::queries::data::Statistics &statistics, sl::queries::algorithms::DistanceType distance_type) {
    for (std::multimap<int64_t, Experiment>::iterator it = experiments.begin(); it != experiments.end(); it++) {
        if (it->second.input_p_size_ == input_q_size && it->second.experiment_name_ == experiment_name && it->second.distance_type_ == distance_type) {
            it->second.statistics_.push_back(statistics);
            return;
        }
    }
}

BASELINE_F(SkylineComputation, SingleThreadBruteForce, InitFromBinaryFileFixture, 5, 10) {
    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type);
    AddStatistics(wq.GetInputP().GetPoints().size(), "SingleThreadBruteForce", statistics, distance_type);
}

BENCHMARK_F(SkylineComputation, SingleThreadBruteForceDiscarting, InitFromBinaryFileFixture, 5, 10) {
    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING, distance_type);
    AddStatistics(wq.GetInputP().GetPoints().size(), "SingleThreadBruteForceDiscarting", statistics, distance_type);
}

BENCHMARK_F(SkylineComputation, SingleThreadSorting, InitFromBinaryFileFixture, 5, 10) {
    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type);
    AddStatistics(wq.GetInputP().GetPoints().size(), "SingleThreadSorting", statistics, distance_type);
}

BENCHMARK_F(SkylineComputation, MultiThreadBruteForce, InitFromBinaryFileFixture, 5, 10) {
    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, distance_type);
    AddStatistics(wq.GetInputP().GetPoints().size(), "MultiThreadBruteForce", statistics, distance_type);
}

//BENCHMARK_F(SkylineComputation, MultiThreadSorting, InitFromBinaryFileFixture, 5, 10) {
//    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, distance_type);
//    AddStatistics(wq.GetInputP().GetPoints().size(), "MultiThreadSorting", statistics, distance_type);
//}

BENCHMARK_F(SkylineComputation, GPUBruteForce, InitFromBinaryFileFixture, 5, 10) {
    sl::queries::data::Statistics statistics = wq.RunAlgorithm(sl::queries::WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type);
    AddStatistics(wq.GetInputP().GetPoints().size(), "GPUBruteForce", statistics, distance_type);
}
