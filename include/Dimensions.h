
#ifndef Dimensions_H_
#define Dimensions_H_
#include <iostream>

struct Dimensions {
    public:
    int dimensions;
    int channels;
    int rows;
    int columns;

    const std::vector<int> vector() const { return std::vector<int>{dimensions, channels, rows, columns};}

    friend std::ostream& operator << (std::ostream &os, const Dimensions &dimension) { 
        os<<dimension.dimensions<<","<<dimension.channels<<","<<dimension.rows<<","<<dimension.columns<<"\n";
        return os;
    }
};

#endif /*Dimensions_H_*/