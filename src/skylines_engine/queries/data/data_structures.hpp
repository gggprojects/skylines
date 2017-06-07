#ifndef SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
#define SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP

namespace sf { namespace queries {namespace data {
    struct Point {
        float x_;
        float y_;
    };

    struct WeightedPoint {
        Point point_;
        float weight_;
    };
}}}

#endif // !SKYLINES_QUERIES_DATA_DATA_STRUCTURES_HPP
