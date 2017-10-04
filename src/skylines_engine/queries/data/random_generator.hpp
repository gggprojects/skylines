
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
            UniformRandomGenerator(std::random_device()()) {
        }

        UniformIntRandomGenerator(unsigned int seed) :
            UniformRandomGenerator(seed) {
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
            UniformRandomGenerator(seed),
            dis(0, 1) {
        }

        double Next() final {
            return dis(gen);
        }
    private:
        std::uniform_real_distribution<> dis;
    };
}}}
#endif
