#include "Neural_Layer.h"

class Maxpool_Layer: public Neural_Layer {
private:
    int _kernel_size;
    int _stride;
    std::vector<unsigned int> _maxpooledIndexes;

public:
    Maxpool_Layer(int kernel_size, int stride);
    Maxpool_Layer(const Maxpool_Layer &maxpool_layer) = delete;
    Maxpool_Layer& operator=(const Maxpool_Layer &maxpool_layer) = delete;
    Maxpool_Layer(Maxpool_Layer &&maxpool_layer) noexcept;
    Maxpool_Layer &operator=(Maxpool_Layer &&maxpool_layer) noexcept;
    ~Maxpool_Layer();

    void PrintMetaData() override;
    void Build(Neural_Layer const* previousLayer) override;
    Tensor const* ForwardPropogate(Tensor const* input) override;
    Tensor* Backpropogate(Tensor* gradient) override;
    void Training(bool train) override;
    
    friend class Network_Saver;
};