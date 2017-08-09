
#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr), algorithms_(6) {

        algorithms_[AlgorithmType::SINGLE_THREAD_BRUTE_FORCE] = std::make_shared<algorithms::SingleThreadBruteForce>(input_p_, input_q_);
        algorithms_[AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING] = std::make_shared<algorithms::SingleThreadBruteForceDiscarting>(input_p_, input_q_);
        algorithms_[AlgorithmType::SINGLE_THREAD_SORTING] = std::make_shared<algorithms::SingleThreadSorting>(input_p_, input_q_);
        algorithms_[AlgorithmType::MULTI_THREAD_BRUTE_FORCE] = std::make_shared<algorithms::MultiThreadBruteForce>(input_p_, input_q_);
        algorithms_[AlgorithmType::MULTI_THREAD_SORTING] = std::make_shared < algorithms::MultiThreadSorting > (input_p_, input_q_);
        algorithms_[AlgorithmType::GPU_BRUTE_FORCE] = std::make_shared<algorithms::GPUBruteForce>(input_p_, input_q_);
    }

    void WeightedQuery::InitRandom(size_t num_points_p, size_t num_points_q) {
        input_p_.InitRandom(num_points_p);
        input_q_.InitRandom(num_points_q);
    }

    void WeightedQuery::Render() const {
        glColor3f(1, 0, 0);
        glPointSize(3);
        input_p_.Render();

        glColor3f(0, 1, 0);
        glPointSize(3);
        input_q_.Render();

        glColor3f(0, 0, 1);
        glPointSize(6);
        output_.Render();
    }

    void WeightedQuery::RunSingleThreadBruteForce() {
        RunAlgorithm(AlgorithmType::SINGLE_THREAD_BRUTE_FORCE);
    }

    void WeightedQuery::RunSingleThreadBruteForceDiscarting() {
        RunAlgorithm(AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING);
    }

    void WeightedQuery::RunSingleThreadSorting() {
        RunAlgorithm(AlgorithmType::SINGLE_THREAD_SORTING);
    }

    void WeightedQuery::RunMultiThreadBruteForce() {
        RunAlgorithm(AlgorithmType::MULTI_THREAD_BRUTE_FORCE);
    }

    void WeightedQuery::RunMultiThreadSorting() {
        RunAlgorithm(AlgorithmType::MULTI_THREAD_SORTING);
    }

    void WeightedQuery::RunGPUBruteForce() {
        RunAlgorithm(AlgorithmType::GPU_BRUTE_FORCE);
    }

    void WeightedQuery::RunAlgorithm(AlgorithmType type) {
        algorithms_[type]->Run(&output_);
    }

}}
