#include "Neural_Layer.h"

class Convolution_Layer: public Neural_Layer {
    int _stride;
    int _kernels;
    int _kernel_size;
    public:
        Convolution_Layer(int kernels, int kernel_size, int stride);
        Convolution_Layer(const Convolution_Layer &convolution_layer) = delete;
        Convolution_Layer& operator=(const Convolution_Layer &convolution_layer) = delete;
        Convolution_Layer(Convolution_Layer &&convolution_layer) noexcept;
        Convolution_Layer& operator=(Convolution_Layer &&convolution_layer) noexcept;
        ~Convolution_Layer();

        void PrintMetaData() override;
        void Build(Neural_Layer const* previousLayer) override;
        Tensor const* ForwardPropogate(Tensor const* input) override;
        Tensor* Backpropogate(Tensor* gradient) override;
        void SetBatchDimensions(int batch_size) override;

        friend class Network_Saver;
};