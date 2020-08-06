#include "Neural_Layer.h"

class Output_Layer: public Neural_Layer {

private:
    std::unique_ptr<float> _error;
    float _loss;
    auto ReturnLossFunction() -> float (*)(const float*, float*, int, int);
    int _batchesInIteration = 1;
public:
    Output_Layer(std::vector<int> dimensions, Activation_Function af);
    ~Output_Layer();
    Output_Layer(const Output_Layer &output_layer) = delete;
    Output_Layer& operator=(const Output_Layer &output_layer) = delete;
    Output_Layer(Output_Layer &&output_layer);
    Output_Layer& operator=(Output_Layer &&output_layer);

    void Build(std::shared_ptr<Neural_Layer> previous_layer);
    void PrintMetaData() override;
    void Training(bool train) override;
    void ForwardPropogate() override;
    void CalculateError(float **target, float regularization);
    void ResetLoss() override;
    float ReturnLoss() const override;

    void PrintFinalResults();
    void PrintError();
    void Backpropogate();

};