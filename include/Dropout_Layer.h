#include "Neural_Layer.h"


class Dropout_Layer: public Neural_Layer {
private:
    std::vector<float> _mask;
    float _percentage;
    bool _training = false;
    std::mt19937 _generator;
public:
    Dropout_Layer(Dimensions dimensions, float percentDropped);
    Dropout_Layer(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer& operator=(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer(Dropout_Layer &&dropout_layer) = default;
    Dropout_Layer &operator=(Dropout_Layer &&dropout_layer) = default;
    ~Dropout_Layer() = default;

    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;

    float returnPercentageDropped() const;
};

