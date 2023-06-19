#include "Neural_Layer.h"

class Fully_Connected_Layer: public Neural_Layer {
    public:
        Fully_Connected_Layer(Dimensions dimensions, Activation_Function af);
        Fully_Connected_Layer(const Fully_Connected_Layer &fully_connected_layer) = delete;
        Fully_Connected_Layer& operator=(const Fully_Connected_Layer &fully_connected_layer) = delete;
        Fully_Connected_Layer(Fully_Connected_Layer &&fully_connected_layer) noexcept;
        Fully_Connected_Layer& operator=(Fully_Connected_Layer &&fully_connected_layer) noexcept;
        ~Fully_Connected_Layer();

        void PrintMetaData() override;
        void Build(Neural_Layer const* previousLayer) override;
        Tensor const* ForwardPropogate(Tensor const* input) override;
        Tensor* Backpropogate(Tensor* gradient) override;
};