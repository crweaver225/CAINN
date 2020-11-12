#include "Neural_Layer.h"

class Embedding_Layer: public Neural_Layer {
public:
    Embedding_Layer(std::vector<int> dimensions);
    Embedding_Layer(const Embedding_Layer &embedding_layer) = delete;
    Embedding_Layer& operator=(const Embedding_Layer &embedding_layer) = delete;
    Embedding_Layer(Embedding_Layer &&embedding_layer);
    Embedding_Layer &operator=(Embedding_Layer &&embedding_layer);
    ~Embedding_Layer();

    void PrintMetaData() override;
    void Build(std::shared_ptr<Neural_Layer> previous_layer) override ;
    void ForwardPropogate() override;
    void Backpropogate() override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;
};