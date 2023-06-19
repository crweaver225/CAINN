#include "Neural_Layer.h"

class Input_layer: public Neural_Layer {
private:
    std::unique_ptr<float> _inputArray;
public:
    Input_layer(Dimensions dimensions);
    ~Input_layer();
    Input_layer(const Input_layer &input_layer) = delete;
    Input_layer& operator=(const Input_layer &input_layer) = delete;
    Input_layer(Input_layer &&input_layer) noexcept;
    Input_layer& operator=(Input_layer &&input_layer) noexcept;

    void PrintMetaData() override;
    void Build(Neural_Layer const *previousLayer) override;
    Tensor const* AddInput(float *input);
    Tensor const* AddInputInBatches(const int dimensions, float **input);
    Tensor const* ForwardPropogate(Tensor const*) override;
    Tensor *  Backpropogate(Tensor* gradient){ return _gradient.get(); }
    void SetBatchDimensions(int batch_size) override;
};