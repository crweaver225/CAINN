#include "Neural_Layer.h"

class Maxpool_Layer: public Neural_Layer {
private:
    int _filters;
    int _filterSize;
    int _stride;
    std::vector<int> _maxpooledIndexes;
    std::vector<int> _outputDimensions;

public:
    Maxpool_Layer(Dimensions dimensions);
    Maxpool_Layer(const Maxpool_Layer &maxpool_layer) = delete;
    Maxpool_Layer& operator=(const Maxpool_Layer &maxpool_layer) = delete;
    Maxpool_Layer(Maxpool_Layer &&maxpool_layer) noexcept;
    Maxpool_Layer &operator=(Maxpool_Layer &&maxpool_layer) noexcept;
    ~Maxpool_Layer();

    //const std::vector<int>& OutputDimensions() override;
    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;

    int returnFilterSize() const;
    int returnStride() const;
};