#include "Neural_Layer.h"

class Flatten_Layer: public Neural_Layer {
    

public:
    Flatten_Layer();
    Flatten_Layer(const Flatten_Layer &flatten_layer) = delete;
    Flatten_Layer& operator=(const Flatten_Layer &flatten_layer) = delete;
    Flatten_Layer(Flatten_Layer &&flatten_layer);
    Flatten_Layer &operator=(Flatten_Layer &&flatten_layer);
    ~Flatten_Layer();

    void PrintMetaData() override;
    void Build(std::shared_ptr<Neural_Layer> previous_layer) override ;
    void ForwardPropogate() override;
    void Backpropogate() override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;
};