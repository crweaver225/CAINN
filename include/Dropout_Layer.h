#include "Neural_Layer.h"


class Dropout_Layer: public Neural_Layer {
private:
    std::vector<int> _droppedNeurons;
    int _neurons;
    float _percentage;
public:
    Dropout_Layer(Dimensions dimensions, float percentDropped);
    Dropout_Layer(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer& operator=(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer(Dropout_Layer &&dropout_layer) noexcept;
    Dropout_Layer &operator=(Dropout_Layer &&dropout_layer) noexcept;
    ~Dropout_Layer();

    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void SetBatchDimensions(int batch_size) override;
    void randomizeDropped();
    void Training(bool train) override;

    float returnPercentageDropped() const;
};

