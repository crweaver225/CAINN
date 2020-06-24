#include "Neural_Layer.h"

class Input_layer: public Neural_Layer {
public:
    Input_layer(std::vector<int> dimensions);
    ~Input_layer();
    Input_layer(const Input_layer &input_layer);
    Input_layer& operator=(const Input_layer &input_layer);
    Input_layer(Input_layer &&input_layer);
    Input_layer& operator=(Input_layer &&input_layer);

    void printMetaData() override;
    void build(std::shared_ptr<Neural_Layer> previous_layer);
    void addInput(float *input);
    void addInputInBatches(const int dimensions, float **input);
    void forward_propogate(){};
    void backpropogate(){};
    void setBatchDimensions(int batch_size) override;
};