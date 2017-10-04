
#include "queries/weighted.hpp"

#pragma warning(push, 0)
#include <freeglut/GL/freeglut.h>
#pragma warning(pop)

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery() :
        SkylineElement("WeightedQuery", "info"), algorithms_(6) {

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
        glPointSize(4);

        //input_q
        glBegin(GL_POINTS);
        for (const sl::queries::data::WeightedPoint &w_point : input_p_.GetPoints()) {
            glVertex2f(w_point.point_.x_, w_point.point_.y_);
        }
        glEnd();

        //text rendering
        //glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
        //for (const sl::queries::data::WeightedPoint &w_point : input_p_.GetPoints()) {
        //    glRasterPos2f(w_point.point_.x_ + 0.005, w_point.point_.y_ + 0.005);
        //    std::string w_text = std::to_string(w_point.weight_);
        //    w_text = w_text.substr(0, 4);
        //    glutBitmapString(GLUT_BITMAP_HELVETICA_12, reinterpret_cast<unsigned char*>(const_cast<char*>(w_text.data())));
        //}

        //input_p
        glColor3f(0, 1, 0);
        glPointSize(4);
        glBegin(GL_POINTS);
        for (const sl::queries::data::Point &p : input_q_.GetPoints()) {
            glVertex2f(p.x_, p.y_);
        }
        glEnd();

        //output
        glColor3f(0, 0, 1);
        glPointSize(7);
        glBegin(GL_POINTS);
        for (const sl::queries::data::WeightedPoint &w_point : output_.GetPoints()) {
            glVertex2f(w_point.point_.x_, w_point.point_.y_);
        }
        glEnd();
    }

    void WeightedQuery::RunAlgorithm(AlgorithmType type, algorithms::DistanceType distance_type) {
        algorithms_[type]->Run(&output_, distance_type);
    }

}}
