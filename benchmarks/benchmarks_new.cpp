
#include <iostream>
#include <fstream>

#include <Windows.h>

#include "time.hpp"
#include "queries/weighted.hpp"

using namespace sl::queries;

WeightedQuery wq;
HANDLE hConsole;

struct ExperimentStadistics {
    ExperimentStadistics() {
        time_taken_ = std::numeric_limits<long long>::max();
    }

    Stadistics stats_;
    long long time_taken_;
};

struct Experiment {
public:
    Experiment(WeightedQuery::AlgorithmType algorithm_type,
    algorithms::DistanceType distance_type,
    size_t input_p_size,
    size_t input_q_size) {
        algorithm_type_ = algorithm_type;
        distance_type_ = distance_type;
        input_p_size_ = input_p_size;
        input_q_size_ = input_q_size;
    }

    bool operator<(const Experiment &other)  const {
        if (algorithm_type_ != other.algorithm_type_) return algorithm_type_ < other.algorithm_type_;
        if (distance_type_ != other.distance_type_) return distance_type_ < other.distance_type_;
        if (input_p_size_ != other.input_p_size_) return input_p_size_ < other.input_p_size_;
        return input_q_size_ < other.input_q_size_;
    }

    WeightedQuery::AlgorithmType algorithm_type_;
    algorithms::DistanceType distance_type_;
    size_t input_p_size_;
    size_t input_q_size_;
};

std::string GetAlgorithmTypeString(algorithms::DistanceType distance_type) {
    switch (distance_type) {
        case sl::queries::algorithms::DistanceType::Neartest: return "Nearest";
        case sl::queries::algorithms::DistanceType::Furthest: return "Furthest";
        default: return "";
    }
}

std::string GetAlgorithmTypeString(WeightedQuery::AlgorithmType algorithm_type) {
    switch (algorithm_type) {
    case WeightedQuery::SINGLE_THREAD_BRUTE_FORCE: return "STBF";
    case WeightedQuery::SINGLE_THREAD_BRUTE_FORCE_DISCARDING: return "STBFD";
    case WeightedQuery::SINGLE_THREAD_SORTING: return "STS";
    case WeightedQuery::MULTI_THREAD_BRUTE_FORCE_DISCARDING: return "MTBTD";
    case WeightedQuery::GPU_BRUTE_FORCE: return "GPUBF";
    default: return "";
    }
}

ExperimentStadistics Execute(
    WeightedQuery::AlgorithmType algorithm_type,
    algorithms::DistanceType distance_type) {

    ExperimentStadistics min_es;
    const int ITERATIONS = 1;
    for (int i = 0; i < ITERATIONS; i++) {
        ExperimentStadistics current_es;
        current_es.time_taken_ = bachelor::time::measure<>::execution([&]() {
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
    data::UniformIntRandomGenerator irg(1, 10);
    wq.InitRandom(input_p_size, input_q_size, rrg_x, rrg_y, irg);
}

void Compare(const Experiment &baseline_experiment, const Experiment &b, const std::map<Experiment, ExperimentStadistics> &experiments) {
    long long baseline_time = experiments.find(baseline_experiment)->second.time_taken_;
    long long time_b = experiments.find(b)->second.time_taken_;

    float improvement = baseline_time / static_cast<float>(time_b);
    if (improvement > 1) {
        SetConsoleTextAttribute(hConsole, 2);
    } else {
        SetConsoleTextAttribute(hConsole, 4);
    }
    std::cout << '\t' << GetAlgorithmTypeString(b.algorithm_type_) << "\t" << improvement << '\n';
}

void RunAllExperiments(size_t input_p_size, size_t input_q_size, std::map<Experiment, ExperimentStadistics> *experiments) {
    SetConsoleTextAttribute(hConsole, 7);
    algorithms::DistanceType distance_type = algorithms::DistanceType::Neartest;
    LoadData(input_p_size, input_q_size);

    std::cout << GetAlgorithmTypeString(distance_type) << '\n';
    std::cout << std::to_string(input_p_size) << "x" << std::to_string(input_q_size) << '\n';

    Experiment stbf(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type, input_p_size, input_q_size);
    Experiment stbfd(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, distance_type, input_p_size, input_q_size);
    Experiment mtbfd(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, distance_type, input_p_size, input_q_size);
    Experiment sts(WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type, input_p_size, input_q_size);
    Experiment gpubf(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type, input_p_size, input_q_size);

    //baseline
    experiments->insert(std::make_pair(stbf, Execute(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type)));
    std::cout << '\t' << GetAlgorithmTypeString(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE) << "\t1x\n";

    experiments->insert(std::make_pair(stbfd, Execute(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, distance_type)));
    Compare(stbf, stbfd, *experiments);

    experiments->insert(std::make_pair(mtbfd, Execute(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, distance_type)));
    Compare(stbf, mtbfd, *experiments);

    experiments->insert(std::make_pair(sts, Execute(WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type)));
    Compare(stbf, sts, *experiments);

    //experiments->insert(std::make_pair(gpubf, Execute(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type)));
    //Compare(stbf, gpubf, *experiments);
    SetConsoleTextAttribute(hConsole, 7);
}

void writeToCSV(
    const std::string &filename,
    const algorithms::DistanceType distance_type,
    const std::map<Experiment, ExperimentStadistics> &experiments,
    std::function<std::string(const std::pair<Experiment, ExperimentStadistics> &es)> f) {

    std::ofstream file(filename);
    file << "Input size;Single thread;Discarting;Multi thread discarding;Sorting;GPU\n";
    for (const std::pair<Experiment, ExperimentStadistics> &kvp : experiments) {
        file << kvp.first.input_p_size_ << 'x' << kvp.first.input_q_size_ << ';';
        file << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_))) << ';';
        file << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_))) << ';';
        file << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::MULTI_THREAD_BRUTE_FORCE_DISCARDING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_))) << ';';
        file << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_SORTING, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_))) << ';';
        //file << f(*experiments.find(Experiment(WeightedQuery::AlgorithmType::GPU_BRUTE_FORCE, distance_type, kvp.first.input_p_size_, kvp.first.input_q_size_))) << ';';
        file << '\n';
    }
    file.close();
}

int main() {

    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    std::map<Experiment, ExperimentStadistics> experiments;

    //RunAllExperiments(2000, 100, &experiments);
    //RunAllExperiments(5000, 100, &experiments);
    //RunAllExperiments(10000, 100, &experiments);
    //RunAllExperiments(20000, 100, &experiments);
    //RunAllExperiments(50000, 100, &experiments);
    //RunAllExperiments(75000, 100, &experiments);
    //RunAllExperiments(100000, 100, &experiments);

    RunAllExperiments(100, 100, &experiments);
    RunAllExperiments(200, 100, &experiments);
    RunAllExperiments(300, 100, &experiments);
    RunAllExperiments(400, 100, &experiments);
    RunAllExperiments(500, 100, &experiments);

    writeToCSV("nearest-running-time.csv", algorithms::DistanceType::Neartest, experiments, [](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        return std::to_string(es.second.time_taken_);
    });

    writeToCSV("nearest-improvement.csv", algorithms::DistanceType::Neartest, experiments, [&](const std::pair<Experiment, ExperimentStadistics> &es) -> std::string {
        long long baseline_time = experiments.find(Experiment(WeightedQuery::AlgorithmType::SINGLE_THREAD_BRUTE_FORCE, es.first.distance_type_, es.first.input_p_size_, es.first.input_q_size_))->second.time_taken_;
        long long time_b = es.second.time_taken_;
        return std::to_string(baseline_time / static_cast<float>(time_b));
    });
}
