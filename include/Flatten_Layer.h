#include "Neural_Layer.h"

class Flatten_Layer: public Neural_Layer {
    

public:
    Flatten_Layer();
    Flatten_Layer(const Flatten_Layer &flatten_layer) = delete;
    Flatten_Layer& operator=(const Flatten_Layer &flatten_layer) = delete;
    Flatten_Layer(Flatten_Layer &&flatten_layer) noexcept;
    Flatten_Layer &operator=(Flatten_Layer &&flatten_layer) noexcept;
    ~Flatten_Layer();

    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;
};