
#include <iostream>
#include <fstream>
#include <iomanip>

#include <Windows.h>

#include "common/time.hpp"
#include "queries/weighted.hpp"

using namespace sl::queries;

WeightedQuery wq;
HANDLE hConsole;

struct ExperimentStadistics {
    ExperimentStadistics() {
        time_taken_ = std::numeric_limits<long long>::max();
    }

    data::Statistics stats_;
    long long time_taken_;
};

struct Experiment {
public:
    Experiment(WeightedQuery::AlgorithmType algorithm_type,
    algorithms::DistanceType distance_type,
    size_t input_p_size,
    size_t input_q_size,
    size_t top_k) {
        algorithm_type_ = algorithm_type;
        distance_type_ = distance_type;
        input_p_size_ = input_p_size;
        input_q_size_ = input_q_size;
        top_k_ = top_k;
    }

    bool operator<(const Experiment &other)  const {
        if (algorithm_type_ != other.algorithm_type_) return algorithm_type_ < other.algorithm_type_;
        if (distance_type_ != other.distance_type_) return distance_type_ < other.distance_type_;
        if (input_p_size_ != other.input_p_size_) return input_p_size_ < other.input_p_size_;
        if (input_q_size_ != other.input_q_size_) return input_q_size_ < other.input_q_size_;
        return top_k_ < other.top_k_;
    }

    WeightedQuery::AlgorithmType algorithm_type_;
    algorithms::DistanceType distance_type_;
    size_t input_p_size_;
    size_t input_q_size_;
    size_t top_k_;
};

std::string GetAlgorithmTypeString(algorithms::DistanceType distance_type) {
    switch (distance_type) {
        case sl::queries::algorithms::DistanceType::Nearest: return "Nearest";
        case sl::queries::algorithms::DistanceType::Furthest: return "Furthest";
        default: return "";
    }
}

std::string GetAlgorithmTypeString(WeightedQuery::AlgorithmType algorithm_type, bool is_top_k) {
    switch (algorithm_type) {
        case WeightedQuery::SINGLE_THREAD_BRUTE_FORCE: return is_top_k ? "STBF-TopK" : "STBF";
        case WeightedQuery::SINGLE_THREAD_BRUTE_FORCE_DISCARDING: return is_top_k ? "STBFD-TopK" : "STBFD";
        case WeightedQuery::MULTI_THREAD_BRUTE_FORCE: return is_top_k ? "MTBF-TopK" : "MTBF";
        case WeightedQuery::MULTI_THREAD_BRUTE_FORCE_DISCARDING: return is_top_k ? "MTBFD-TopK" : "MTBFD";
        case WeightedQuery::SINGLE_THREAD_SORTING: return is_top_k ? "STS-TopK" : "STS";
        case WeightedQuery::MULTI_THREAD_SORTING: return is_top_k ? "MTS-TopK" : "MTS";
        case WeightedQuery::GPU_BRUTE_FORCE: return is_top_k ? "GPUBF-TopK" : "GPUBF";
        case WeightedQuery::GPU_BRUTE_FORCE_DISCARTING: return is_top_k ? "GPUBFD-TopK" : "GPUBFD";
        default: return "";
    }
}

int ChooseIterations(size_t input_p_size) {
    if (input_p_size == 2000) return 50;
    if (input_p_size == 5000) return 20;
    if (input_p_size == 10000) return 10;
    if (input_p_size == 20000) return 5;
    if (input_p_size == 50000) return 2;
    if (input_p_size == 75000) return 1;
    if (input_p_size == 100000) return 1;
    return 1;
}

ExperimentStadistics Execute(
    WeightedQuery::AlgorithmType algorithm_type,
    algorithms::DistanceType distance_type) {

    int ITERATIONS = ChooseIterations(wq.GetInputP().GetPoints().size());

    ExperimentStadistics min_es;
    for (int i = 0; i < ITERATIONS; i++) {
        ExperimentStadistics current_es;
        current_es.time_taken_ = sl::time::measure<>::execution([&]() {
            current_es.stats_ = wq.RunAlgorithm(algorithm_type, distance_type);
        });

        if (current_es.time_taken_ < min_es.time_taken_) {
            min_es.time_taken_ = current_es.time_taken_;
            min_es.stats_ = current_es.stats_;
        }
    }
    return min_es;
}

void LoadData(size_t input_p_size, size_t input_q_size) {
    data::UniformRealRandomGenerator rrg_x(0., 1.);
    data::UniformRealRandomGenerator rrg_y(0., 1.);
    //data::UniformIntRandomGenerator irg(1, 1); // big output
    data::UniformIntRandomGenerator irg(1, 10); // small output
    wq.InitRandom(input_p_size, input_q_size, rrg_x, rrg_y, irg);
}

void Compare(const Experiment &baseline_experiment, const Experiment &b, const std::map<Experiment, ExperimentStadistics> &experiments, size_t top_k) {
    long long baseline_time = experiments.find(baseline_experiment)->second.time_taken_;
    const ExperimentStadistics &experiment_statistics = experiments.find(b)->second;

    float improvement = baseline_time / static_cast<float>(experiment_statistics.time_taken_);
    if (improvement >= 1) {
        SetConsoleTextAttribute(hConsole, 2);
    } else {
        SetConsoleTextAttribute(hConsole, 4);
    }
    std::cout << '\t' << GetAlgorithmTypeString(b.algorithm_type_, top_k != b.input_p_size_) << '\t' << improvement << "\tTime Taken: " << experiment_statistics.time_taken_ << "(ms)\tComparisons: " << experiment_statistics.stats_.num_comparisions_ << "\tOutputsize: " << experiment_statistics.stats_.output_size_ << '\n';
}

bool CheckOuput(sl::queries::NonConstData<sl::queries::data::WeightedPoint> &a, sl::queries::NonConstData<sl::queries::data::WeightedPoint> &b, int line) { // we pass a copy
    auto sorting_function = [](const sl::queries::data::WeightedPoint &a, const sl::queries::data::WeightedPoint &b) -> bool {
        if (a.point_.x_ == b.point_.x_) return a.point_.y_ < b.point_.y_;
        else return a.point_.x_ < b.point_.x_;
    };

    std::sort(a.Points().begin(), a.Points().end(), sorting_function);
    std::sort(b.Points().begin(), b.Points().end(), sorting_function);

    if (a.Points().size() != b.Points().size()) {
        std::cerr << "points are different at line: "<< line;
        return false;
    }
    bool equal = std::equal(a.Points().begin(), a.Points().end(), b.Points().begin());
    if (!equal) {
        std::cerr << "points are different at line: " << line;
    }

    return equal;
}

void RunExperiment(
    size_t input_p_size, size_t input_q_size, std::map<Experiment, ExperimentStadistics> *experiments, algorithms::DistanceType distance_type, size_t top_k,
    const Experiment &stbf,
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> &output_baseline,
    WeightedQuery::AlgorithmType algo, int line) {
    Experiment new_experiment(algo, distance_type, input_p_size, input_q_size, top_k);
    experiments->insert(std::make_pair(new_experiment, Execute(algo, distance_type)));
    Compare(stbf, new_experiment, *experiments, top_k);
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> new_ouput = wq.GetOuputCopy();
    CheckOuput(output_baseline, new_ouput, line);
    wq.ClearOutput();
}

void RunExperiments(size_t input_p_size, size_t input_q_size, std::map<Experiment, ExperimentStadistics> *experiments, algorithms::DistanceType distance_type, size_t top_k) {
    SetConsoleTextAttribute(hConsole, 7);
    std::cout << '\n' << GetAlgorithmTypeString(distance_type) << '\n';
    std::cout << std::to_string(input_p_size) << "x" << std::to_string(input_q_size) << '\n';

    wq.SetTopK(top_k);

    //baseline
    Experiment stbf(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type, input_p_size, input_q_size, top_k);
    experiments->insert(std::make_pair(stbf, Execute(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type)));
    const ExperimentStadistics &experiment_statistics = experiments->find(stbf)->second;
    std::cout << std::fixed << std::setprecision(3) << '\t' << GetAlgorithmTypeString(stbf.algorithm_type_, top_k != stbf.input_p_size_) << "\t1x\tTime Taken: " << experiment_statistics.time_taken_ << "(ms)\tComparisons: " << experiment_statistics.stats_.num_comparisions_ << "\tOutputsize: " << experiment_statistics.stats_.output_size_ << '\n';
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> output_baseline = wq.GetOuputCopy();
    sl::queries::NonConstData<sl::queries::data::WeightedPoint> new_ouput;

    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, __LINE__);
    RunExperiment(input_p_size, input_q_size, experiments, distance_type, top_k, stbf, output_baseline, WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE_DISCARTING, __LINE__);

    SetConsoleTextAttribute(hConsole, 7);
}

void RunAllExperiments(size_t input_p_size, size_t input_q_size, std::map<Experiment, ExperimentStadistics> *experiments, size_t top_k) {
    LoadData(input_p_size, input_q_size);
    RunExperiments(input_p_size, input_q_size, experiments, algorithms::DistanceType::Nearest, input_p_size);
    RunExperiments(input_p_size, input_q_size, experiments, algorithms::DistanceType::Furthest, input_p_size);

    RunExperiments(input_p_size, input_q_size, experiments, algorithms::DistanceType::Nearest, top_k);
    RunExperiments(input_p_size, input_q_size, experiments, algorithms::DistanceType::Furthest, top_k);
}

void writeToCSV(
    const std::string &filename,
    const std::map<Experiment, ExperimentStadistics> &experiments,
    std::function<std::string(const std::pair<Experiment, ExperimentStadistics> &es)> f) {
    {
        algorithms::DistanceType distance_type = algorithms::DistanceType::Nearest;
        std::ofstream file("nearest-" + filename + ".csv");
        std::ofstream file_top_k("nearest-" + filename + "-top-k.csv");
        file << "Input size,Single thread BF,Single thread BF Discarting,Multi thread BF,Multi thread BF discarding,Single thread sorting,Multi thread sorting,GPU BF,GPU BF Discarting\n";
        file_top_k << "Input size,Single thread BF,Single thread BF Discarting,Multi thread BF,Multi thread BF discarding,Single thread sorting,Multi thread sorting,GPU BF,GPU BF Discarting\n";
        for (const std::pair<Experiment, ExperimentStadistics> &kvp : experiments) {
            if (kvp.first.algorithm_type_ == WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE && kvp.first.distance_type_ == distance_type) {

                std::ofstream &ff = kvp.first.input_p_size_ == kvp.first.top_k_ ? file : file_top_k;
                ff << kvp.first.input_p_size_ << 'x' << kvp.first.input_q_size_ << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE_DISCARTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_)));
                ff << '\n';
            }
        }
        file.close();
        file_top_k.close();
    }
    {
        algorithms::DistanceType distance_type = algorithms::DistanceType::Furthest;
        std::ofstream file("furthest-" + filename + ".csv");
        std::ofstream file_top_k("furthest-" + filename + "-top-k.csv");
        file << "Input size,Single thread BF,Single thread BF Discarting,Multi thread BF,Multi thread BF discarding,Single thread sorting,Multi thread sorting,GPU BF,GPU BF Discarting\n";
        file_top_k << "Input size,Single thread BF,Single thread BF Discarting,Multi thread BF,Multi thread BF discarding,Single thread sorting,Multi thread sorting,GPU BF,GPU BF Discarting\n";
        for (const std::pair<Experiment, ExperimentStadistics> &kvp : experiments) {
            if (kvp.first.algorithm_type_ == WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE && kvp.first.distance_type_ == distance_type) {

                std::ofstream &ff = kvp.first.input_p_size_ == kvp.first.top_k_ ? file : file_top_k;

                ff << kvp.first.input_p_size_ << 'x' << kvp.first.input_q_size_ << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_SORTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_))) << ',';
                ff << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE_DISCARTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_, kvp.first.top_k_)));
            }
        }
        file.close();
        file_top_k.close();
    }
}

void writeFiles(const std::map<Experiment, ExperimentStadistics> &experiments) {

    //running times
    writeToCSV("running-time", experiments, [](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        return std::to_string(es.second.time_taken_);
    });

    //improvement
    writeToCSV("improvement", experiments, [&](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        if (es.first.algorithm_type_ == WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE) {
            return "1";
        }
        long long baseline_time = experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, es.first.distance_type_, es.first.input_p_size_, es.first.input_q_size_, es.first.top_k_))->second.time_taken_;
        long long time_b = es.second.time_taken_;
        return std::to_string(baseline_time / static_cast<float>(time_b));
    });

    //num-comparisons
    writeToCSV("num-comparisons", experiments, [](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        return std::to_string(es.second.stats_.num_comparisions_);
    });

    //output size
    writeToCSV("output-size", experiments, [](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        return std::to_string(es.second.stats_.output_size_);
    });
}

int main() {

    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);


    std::map<Experiment, ExperimentStadistics> experiments;

    RunAllExperiments(2000, 100, &experiments, 10);
    RunAllExperiments(5000, 100, &experiments, 10);
    RunAllExperiments(10000, 100, &experiments, 20);
    RunAllExperiments(20000, 100, &experiments, 20);
    RunAllExperiments(50000, 100, &experiments, 50);
    RunAllExperiments(75000, 100, &experiments, 75);
    RunAllExperiments(100000, 100, &experiments, 100);
    writeFiles(experiments);

}
