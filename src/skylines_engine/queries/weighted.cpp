
#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr), algorithms_(6) {

        algorithms_[AlgorithmType::SINGLE_THREAD_BRUTE_FORCE] = std::make_shared<algorithms::SingleThreadBruteForce>(error_ptr, input_p_, input_q_);
        algorithms_[AlgorithmType::SINGLE_THREAD_BRUTE_FORCE_DISCARTING] = std::make_shared<algorithms::SingleThreadBruteForceDiscarting>(error_ptr, input_p_, input_q_);
        algorithms_[AlgorithmType::SINGLE_THREAD_SORTING] = std::make_shared<algorithms::SingleThreadSorting>(error_ptr, input_p_, input_q_);
        algorithms_[AlgorithmType::MULTI_THREAD_BRUTE_FORCE] = std::make_shared<algorithms::MultiThreadBruteForce>(error_ptr, input_p_, input_q_);
        algorithms_[AlgorithmType::MULTI_THREAD_SORTING] = std::make_shared < algorithms::MultiThreadSorting > (error_ptr, input_p_, input_q_);
        algorithms_[AlgorithmType::GPU_BRUTE_FORCE] = std::make_shared<algorithms::GPUBruteForce>(error_ptr, input_p_, input_q_);
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

    void WeightedQuery::RunAlgorithm(AlgorithmType type) {
        algorithms_[type]->Run(&output_);
    }

}}
