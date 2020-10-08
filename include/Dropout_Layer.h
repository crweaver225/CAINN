#include "Neural_Layer.h"


class Dropout_Layer: public Neural_Layer {
private:
    std::vector<int> _droppedNeurons;
    int _neurons;
    float _percentage;
public:
    Dropout_Layer(std::vector<int> dimensions, float percentDropped);
    Dropout_Layer(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer& operator=(const Dropout_Layer &dropout_layer) = delete;
    Dropout_Layer(Dropout_Layer &&dropout_layer);
    Dropout_Layer &operator=(Dropout_Layer &&dropout_layer);
    ~Dropout_Layer();

    void PrintMetaData() override;
    void Build(std::shared_ptr<Neural_Layer> previous_layer) override ;
    void ForwardPropogate() override;
    void Backpropogate() override;
    void SetBatchDimensions(int batch_size) override;
    void randomizeDropped();
    void Training(bool train) override;
};
