#include "Neural_Layer.h"

class Output_Layer: public Neural_Layer {

private:
    std::unique_ptr<Tensor> _error;
    float _loss;
    auto ReturnLossFunction() -> float (*)(const float*, float*, int, int);
    int _batchesInIteration = 1;
public:
    Output_Layer(Dimensions dimensions, Activation_Function af);
    ~Output_Layer();
    Output_Layer(const Output_Layer &output_layer) = delete;
    Output_Layer& operator=(const Output_Layer &output_layer) = delete;
    Output_Layer(Output_Layer &&output_layer) noexcept;
    Output_Layer& operator=(Output_Layer &&output_layer) noexcept;

    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;

    void SetLossFunction(Loss loss);
    void PrintMetaData() override;
    void Training(bool train) override;
    void CalculateError(float **target, float regularization);
    void ResetLoss();
    float ReturnLoss() const;
    Tensor* ReturnError() const;
    void SetActiveDimensions(int batch_size) override;

    void PrintFinalResults();
    void PrintError();
};