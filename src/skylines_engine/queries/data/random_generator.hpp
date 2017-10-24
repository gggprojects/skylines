
#ifndef SKYLINES_QUERIES_DATA_RANDOM_GENERATOR_HPP
#define SKYLINES_QUERIES_DATA_RANDOM_GENERATOR_HPP

#include <random>

namespace sl { namespace queries { namespace data {
    template<class T>
    class UniformRandomGenerator {
    public:
        UniformRandomGenerator(unsigned int seed) : gen(seed) {
        }

        virtual T Next() = 0;
    protected:
        std::mt19937 gen; //Standard mersenne_twister_engine
    };

    class UniformIntRandomGenerator : public UniformRandomGenerator<int> {
    public:
        UniformIntRandomGenerator() :
            UniformIntRandomGenerator(std::random_device()()) {
        }

        UniformIntRandomGenerator(unsigned int seed) :
            UniformIntRandomGenerator(seed, 1, 1) {
        }

        UniformIntRandomGenerator(int min, int max) :
            UniformIntRandomGenerator(std::random_device()(), min, max) {
        }

        UniformIntRandomGenerator(unsigned int seed, int min, int max) :
            UniformRandomGenerator(seed) {
            dis.param(std::uniform_int_distribution<int>::param_type(min, max));
        }

        void SetRange(int min, int max) {
            dis.param(std::uniform_int_distribution<int>::param_type(min, max));
        }

        int Next() final {
            return dis(gen);
        }
    private:
        std::uniform_int_distribution<> dis;
    };

    class UniformRealRandomGenerator : public UniformRandomGenerator<double> {
    public:
        UniformRealRandomGenerator() :
            UniformRandomGenerator(std::random_device()()) {
        }

        UniformRealRandomGenerator(unsigned int seed) :
            UniformRealRandomGenerator(seed, 0., 1.) {
        }

        UniformRealRandomGenerator(double min, double max) :
            UniformRealRandomGenerator(std::random_device()(), min, max) {
        }

        UniformRealRandomGenerator(unsigned int seed, double min, double max) :
            UniformRandomGenerator(seed),
            dis(min, max) {
        }

        double Next() final {
            return dis(gen);
        }
    private:
        std::uniform_real_distribution<> dis;
    };
}}}
#endif
