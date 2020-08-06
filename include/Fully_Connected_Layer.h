#include "Neural_Layer.h"

class Fully_Connected_Layer: public Neural_Layer {
public:
    Fully_Connected_Layer(std::vector<int> dimensions, Activation_Function af);
    Fully_Connected_Layer(const Fully_Connected_Layer &fully_connected_layer) = delete;
    Fully_Connected_Layer& operator=(const Fully_Connected_Layer &fully_connected_layer) = delete;
    Fully_Connected_Layer(Fully_Connected_Layer &&fully_connected_layer);
    Fully_Connected_Layer& operator=(Fully_Connected_Layer &&fully_connected_layer);
    ~Fully_Connected_Layer();

    void PrintMetaData() override;
    void Build(std::shared_ptr<Neural_Layer> previous_layer);
    void ForwardPropogate();
    void Backpropogate();
};