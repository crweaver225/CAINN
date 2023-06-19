#include "Neural_Layer.h"

class Embedding_Layer: public Neural_Layer {
public:
    Embedding_Layer(Dimensions dimensions);
    Embedding_Layer(const Embedding_Layer &embedding_layer) = delete;
    Embedding_Layer& operator=(const Embedding_Layer &embedding_layer) = delete;
    Embedding_Layer(Embedding_Layer &&embedding_layer) noexcept;
    Embedding_Layer &operator=(Embedding_Layer &&embedding_layer) noexcept;
    ~Embedding_Layer();

    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void SetBatchDimensions(int batch_size) override;
    void Training(bool train) override;
};