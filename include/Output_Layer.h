#include "Neural_Layer.h"

class Output_Layer: public Neural_Layer {

private:
    std::unique_ptr<float> error;
   //std::vector<float> error;
    float loss;
public:
    Output_Layer(std::vector<int> dimensions, Activation_Function af);
    ~Output_Layer();
    Output_Layer(const Output_Layer &output_layer);
    Output_Layer& operator=(const Output_Layer &output_layer);
    Output_Layer(Output_Layer &&output_layer);
    Output_Layer& operator=(Output_Layer &&output_layer);

    void build(std::shared_ptr<Neural_Layer> previous_layer);
    void printMetaData() override;
    void training(bool train) override;
    void forward_propogate() override;
    void calculateError(float **target, float regularization);

    void printFinalResults();
    void printError();
    void backpropogate();

};