#include "Neural_Layer.h"

class Input_layer: public Neural_Layer {
private:
    std::unique_ptr<float> _inputArray;
public:
    Input_layer(std::vector<int> dimensions);
    ~Input_layer();
    Input_layer(const Input_layer &input_layer) = delete;
    Input_layer& operator=(const Input_layer &input_layer) = delete;
    Input_layer(Input_layer &&input_layer);
    Input_layer& operator=(Input_layer &&input_layer);

    void PrintMetaData() override;
    void Build(std::shared_ptr<Neural_Layer> previous_layer);
    void AddInput(float *input);
    void AddInputInBatches(const int dimensions, float **input);
    void ForwardPropogate(){};
    void Backpropogate(){};
    void SetBatchDimensions(int batch_size) override;
};