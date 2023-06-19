
#ifndef Dimensions_H_
#define Dimensions_H_

struct Dimensions {
    public:
    int dimensions;
    int channels;
    int rows;
    int columns;

    const std::vector<int> vector() const { return std::vector<int>{dimensions, channels, rows, columns};}
};

#endif /*Dimensions_H_*/