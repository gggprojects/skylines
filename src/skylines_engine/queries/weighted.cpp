#include <algorithm>

#include "queries/weighted.hpp"

namespace sl { namespace queries {
    WeightedQuery::WeightedQuery(error::ThreadErrors_ptr error_ptr) :
        SkylineElement("WeightedQuery", "info", error_ptr) {
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

    bool WeightedQuery::IsEmpty() {
        return input_p_.GetPoints().empty() || input_q_.GetPoints().empty();
    }

    bool WeightedQuery::Init() {
        if (IsEmpty()) return false;
        ClearOutput();
        return true;
    }

    void WeightedQuery::RunSingleThreadBruteForce() {
        if (!Init()) return;
        ComputeSingleThreadBruteForce();
    }

    void WeightedQuery::RunSingleThreadBruteForceDiscarting() {
        if (!Init()) return;
        ComputeSingleThreadBruteForceDiscarting();
    }

    void WeightedQuery::RunSingleThreadSorting() {
        if (!Init()) return;
        ComputeSkylineSingleThreadSorting();
    }

    void WeightedQuery::RunMultiThreadBruteForce() {
        if (!Init()) return;
        ComputeMultiThreadBruteForce();
    }

    void WeightedQuery::RunGPUBruteForce() {
        if (!Init()) return;
        ComputeGPUBruteForce();
    }

    bool IsDominated(const data::WeightedPoint &a, const data::WeightedPoint &b, const std::vector<data::Point> &q) {
        for (const data::Point p_q : q) {
            float a_distance = a.SquaredDistance(p_q);
            float b_distance = b.SquaredDistance(p_q);
            if (a_distance <= b_distance) {
                return false;
            }
        }
        return true;
    }

    void WeightedQuery::ComputeSingleThreadBruteForce() {
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = input_p_.GetPoints().cbegin();
            skyline_candidate != input_p_.GetPoints().cend(); ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator dominator_candidate = input_p_.GetPoints().cbegin();
            bool is_skyline = true;
            while (is_skyline && dominator_candidate != input_p_.GetPoints().cend()) {
                if (skyline_candidate != dominator_candidate) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q_.GetPoints())) {
                        is_skyline = false;
                    }
                }
                ++dominator_candidate;
            }

            if (is_skyline) {
                output_.Add(*skyline_candidate);
            }
        }
    }

    int Dominator(const data::WeightedPoint &a, const data::WeightedPoint &b, const std::vector<data::Point> &q) {
        bool a_is_dominated_by_b = true;
        bool b_is_dominated_by_a = true;
        for (const data::Point p_q : q) {
            float a_distance = a.Distance(p_q);
            float b_distance = b.Distance(p_q);
            if (a_distance <= b_distance) {
                a_is_dominated_by_b = false;
                if (!b_is_dominated_by_a) return -1;
            }
            if (b_distance <= a_distance) {
                b_is_dominated_by_a = false;
                if (!a_is_dominated_by_b) return -1;
            }
        }
        //at this point one domains the other
        if (a_is_dominated_by_b) return 1;
        return 0;
    }

    void WeightedQuery::ComputeSingleThreadBruteForceDiscarting() {
        std::vector<bool> needs_to_be_checked(input_p_.GetPoints().size(), true);

        std::vector<bool>::iterator a_needs_to_be_checked = needs_to_be_checked.begin();
        for (std::vector<data::WeightedPoint>::const_iterator a = input_p_.GetPoints().cbegin(); a != input_p_.GetPoints().cend(); ++a, ++a_needs_to_be_checked) {
            if (*a_needs_to_be_checked) {
                std::vector<bool>::iterator b_needs_to_be_checked = needs_to_be_checked.begin();
                std::vector<data::WeightedPoint>::const_iterator b = input_p_.GetPoints().cbegin();

                bool is_skyline = true;
                while (is_skyline && b != input_p_.GetPoints().cend()) {
                    if (a != b && *b_needs_to_be_checked) {
                        int dominator = Dominator(*a, *b, input_q_.GetPoints());
                        if (dominator == 1) {
                            is_skyline = false;
                        } else if (dominator == 0) {
                            *b_needs_to_be_checked = false;
                        }
                    }
                    ++b;
                    ++b_needs_to_be_checked;
                }

                if (is_skyline) {
                    output_.Add(*a);
                }
            }
        }
    }

    void WeightedQuery::ComputeSkylineSingleThreadSorting() {
        //copy P
        NonConstData<data::WeightedPoint> sorted_input;
        sorted_input = input_p_;

        // sort by the first point in Q
        const data::Point &first_q = input_q_.GetPoints()[0];
        std::sort(sorted_input.Points().begin(), sorted_input.Points().end(), [&first_q](const data::WeightedPoint &a, const data::WeightedPoint &b) {
            return a.SquaredDistance(first_q) < b.SquaredDistance(first_q);
        });

        //the first element is skyline
        output_.Add(sorted_input.GetPoints()[0]);
        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = sorted_input.GetPoints().cbegin() + 1; skyline_candidate != sorted_input.GetPoints().cend(); ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_reverse_iterator skyline_element = output_.GetPoints().crbegin();
            bool is_skyline = true;
            while (is_skyline && skyline_element != output_.Points().rend()) {
                if (skyline_candidate->IsDominated(*skyline_element, input_q_.GetPoints())) {
                    is_skyline = false;
                }
                skyline_element++;
            }
            if (is_skyline) {
                output_.Add(*skyline_candidate);
            }
        }
    }

    void WeightedQuery::ComputeSingleThreadBruteForce(
        std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate,
        std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate) {

        for (std::vector<data::WeightedPoint>::const_iterator skyline_candidate = first_skyline_candidate; skyline_candidate != last_skyline_candidate; ++skyline_candidate) {
            std::vector<data::WeightedPoint>::const_iterator dominator_candidate = input_p_.GetPoints().cbegin();
            bool is_skyline = true;
            while (is_skyline && dominator_candidate != input_p_.GetPoints().cend()) {
                if (skyline_candidate != dominator_candidate) {
                    if (IsDominated(*skyline_candidate, *dominator_candidate, input_q_.GetPoints())) {
                        is_skyline = false;
                    }
                }
                ++dominator_candidate;
            }

            if (is_skyline) {
                output_.SafetyAdd(*skyline_candidate);
            }
        }
    }

    void WeightedQuery::ComputeMultiThreadBruteForce() {
        unsigned int concurent_threads_supported = std::thread::hardware_concurrency();
        size_t num_elements_p = input_p_.GetPoints().size();

        if (concurent_threads_supported > num_elements_p) {
            concurent_threads_supported = num_elements_p;
        }

        size_t chunk = num_elements_p / concurent_threads_supported;
        std::vector<std::thread> workers;
        for (size_t i = 0; i < concurent_threads_supported; i++) {
            workers.emplace_back(
            std::thread ([this, &chunk, i] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (i * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().cbegin() + ((i + 1) * chunk);
                ComputeSingleThreadBruteForce(first_skyline_candidate, last_skyline_candidate);
            }));
        }

        if (input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk) != input_p_.GetPoints().cend()) {
            workers.emplace_back(
                std::thread([this, &chunk, &concurent_threads_supported] {
                std::vector<data::WeightedPoint>::const_iterator first_skyline_candidate = input_p_.GetPoints().cbegin() + (concurent_threads_supported * chunk);
                std::vector<data::WeightedPoint>::const_iterator last_skyline_candidate = input_p_.GetPoints().end();
                ComputeSingleThreadBruteForce(first_skyline_candidate, last_skyline_candidate);
            }));
        }

        for (std::thread &t : workers) {
            t.join();
        }
    }

    void WeightedQuery::ComputeGPUBruteForce() {
    }
}}
