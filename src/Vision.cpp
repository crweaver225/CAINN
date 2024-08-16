
#include "Vision.h"

std::mutex backward_mutex{};

// ### Convolution Methods ###
void Vision::InnerBackward(const Tensor &gradient, const Tensor &kernel, Tensor &_gradient, int stride, const int d, const int n_d) {
  
    int gradient_active_batches = gradient.ReturnActiveDimension(); 
    int gradient_channels = gradient.NumberOfChannels(); 
    int gradient_rows = gradient.NumberOfRows(); 
    int gradient_columns = gradient.NumberOfColumns(); 
    const float* gradient_data = gradient.ReturnData(); 

    int kernel_dimensions = kernel.NumberOfDimensions();
    int kernel_channels = kernel.NumberOfChannels(); 
    int kernel_rows = kernel.NumberOfRows(); 
    int kernel_columns = kernel.NumberOfColumns(); 
    const float* kernel_data = kernel.ReturnData(); 

    int _gradient_active_batches = _gradient.ReturnActiveDimension();
    int _gradient_channels = _gradient.NumberOfChannels(); 
    int _gradient_rows = _gradient.NumberOfRows(); 
    int _gradient_columns = _gradient.NumberOfColumns(); 

    int gradient_dimension_idx = d * (gradient_channels * gradient_rows * gradient_columns);
    int _gradient_dimension_idx = d * (_gradient_channels * _gradient_rows * _gradient_columns);

    for (int dimension_tracker = 0; dimension_tracker < n_d; dimension_tracker++) {
        
        for (int kernel = 0; kernel < kernel_dimensions; kernel++) {

            const int kernel_dimension_idx = kernel * (kernel_channels * kernel_rows * kernel_columns);
            const int gradient_channel_idx = gradient_dimension_idx + (kernel * gradient_rows * gradient_columns);


            for (int kernel_channel = 0; kernel_channel < kernel_channels; kernel_channel++) {
            
                const int kernel_channel_idx = kernel_dimension_idx + (kernel_channel * (kernel_rows * kernel_columns));
                const int _gradient_channel_idx = _gradient_dimension_idx + (kernel_channel * _gradient_rows * _gradient_columns);

                for (int kernel_row = 0; kernel_row < kernel_rows; kernel_row++) {

                    int kernel_row_idx = kernel_channel_idx + (kernel_row * kernel_columns);

                    for (int kernel_column = 0; kernel_column < kernel_columns; kernel_column++) {

                        const int kernel_column_idx = kernel_row_idx + kernel_column;
                        const float kernel_value = kernel_data[kernel_column_idx];
                                                                          
                        for (int gradient_row = 0; gradient_row < gradient_rows; gradient_row ++) {
                            
                            const int gradient_row_idx = gradient_channel_idx + (gradient_row * gradient_columns);
                            const int _gradient_row_idx = (_gradient_channel_idx + ((gradient_row + kernel_row) * _gradient_columns));
                            
                            for (int gradient_column = 0; gradient_column < gradient_columns; gradient_column ++) {
                                
                                const int gradient_column_idx = gradient_row_idx + gradient_column;
                                const int _gradient_column_idx = _gradient_row_idx + gradient_column + kernel_column;
                                
                                _gradient.changeNeuron(_gradient_column_idx, gradient_data[gradient_column_idx] * kernel_value);
                                
                            }
                        }
                    }
                }
            }
        }
        gradient_dimension_idx += (gradient_channels * gradient_rows * gradient_columns);
        _gradient_dimension_idx += (_gradient_channels * _gradient_rows * _gradient_columns);
    }
    
}

void Vision::Backward(const Tensor &gradient, const Tensor &kernel, Tensor &_gradient, Thread_Pool& threadPool, int stride) {
    int activeDimensions = gradient.ReturnActiveDimension();
    const auto processor_count = std::thread::hardware_concurrency();
    if (activeDimensions >= processor_count) {
        
        const int dimensions_per_thread = activeDimensions / processor_count; 
        const int dimensions_per_thread_remainder = activeDimensions % processor_count; 

       for (int i = 0; i < processor_count - 1; i++) { 
            threadPool.enqueue(&Vision::InnerBackward, 
                                    std::ref(gradient), 
                                    std::ref(kernel), 
                                    std::ref(_gradient), 
                                    stride, 
                                    i * dimensions_per_thread, 
                                    dimensions_per_thread);
        }
        threadPool.enqueue(&Vision::InnerBackward, 
                                    std::ref(gradient), 
                                    std::ref(kernel), 
                                    std::ref(_gradient), 
                                    stride, 
                                    (processor_count-1) * dimensions_per_thread, 
                                    dimensions_per_thread + dimensions_per_thread_remainder);
        
        threadPool.wait();

    } else {
        for (int i = 0; i < activeDimensions; i++) {
            Vision::InnerBackward(gradient, kernel, _gradient, stride, i, 1);
        }
    }
    
    _gradient.clipData();
}

void Vision::UpdateKernel(const Tensor &input, Tensor &kernel, const Tensor &gradient, Thread_Pool& threadPool, int stride) {

    int activeDimensions = gradient.ReturnActiveDimension();
    const auto processor_count = std::thread::hardware_concurrency();

    if (activeDimensions >= processor_count) {

        const int dimensions_per_thread = activeDimensions / processor_count; 
        const int dimensions_per_thread_remainder = activeDimensions % processor_count; 

         for (int i = 0; i < processor_count - 1; i++) { 
            threadPool.enqueue(&Vision::InnerUpdateKernel, 
                                    std::ref(input), 
                                    std::ref(kernel), 
                                    std::ref(gradient), 
                                    stride, 
                                    i * dimensions_per_thread, 
                                    dimensions_per_thread);
        }

        threadPool.enqueue(&Vision::InnerUpdateKernel, 
                                    std::ref(input), 
                                    std::ref(kernel), 
                                    std::ref(gradient), 
                                    stride, 
                                    (processor_count-1) * dimensions_per_thread, 
                                    dimensions_per_thread + dimensions_per_thread_remainder);
        
        threadPool.wait();
       
    } else {
        for (int i = 0; i < activeDimensions; i++) {
            Vision::InnerUpdateKernel(input, kernel, gradient, stride, i, 1);
        }
    }
}

void Vision::InnerUpdateKernel(const Tensor &input, Tensor &kernel, const Tensor &gradient, int stride, const int d, const int n_d) {

    int gradient_row_size = gradient.NumberOfRows();
    int gradient_column_size = gradient.NumberOfColumns();
    int gradient_channel_size = gradient.NumberOfChannels();
    const float *gradient_data = gradient.ReturnData();

    int input_channels_size = input.NumberOfChannels();
    int input_rows_size = input.NumberOfRows();
    int input_columns_size = input.NumberOfColumns();
    const float *input_data = input.ReturnData();

    const int active_dimensions = gradient.ReturnActiveDimension();

    Dimensions kernel_dimensions = kernel.dimensions();
 
    int gradient_batch_idx = d * gradient.NumberOfElementsPerTensor();
    int input_batch_idz = d * input.NumberOfElementsPerTensor();

    for (int gradient_channel = 0; gradient_channel < gradient_channel_size; gradient_channel++) {

        int gradient_channel_idx = gradient_batch_idx + (gradient_channel * gradient_row_size * gradient_column_size);
        int kernel_idx = gradient_channel * kernel_dimensions.channels * kernel_dimensions.rows * kernel_dimensions.columns;

        for (int input_channel = 0; input_channel < input_channels_size; input_channel ++) {

            int input_chanel_idx = input_batch_idz + (input_channel * input_rows_size * input_columns_size);
            int kernel_channel_idx = kernel_idx + (input_channel * kernel_dimensions.rows * kernel_dimensions.columns);
            int start_gradient_row = 0;

            while (start_gradient_row + gradient_row_size <=  input_rows_size) {

                int start_gradient_column = 0;
                int kernel_row_idx = kernel_channel_idx + (start_gradient_row  * kernel_dimensions.columns);
                int kernel_column_idx = 0;
    
                while (start_gradient_column + gradient_column_size <= input_columns_size) {

                    float sum = 0.0f;
                    
                    for (int gradient_row = 0; gradient_row < gradient_row_size; gradient_row++) {
                     
                        int gradient_row_idx = gradient_channel_idx + (gradient_row * gradient_column_size);
                        int input_row_idx = input_chanel_idx + ((start_gradient_row + gradient_row) * input_columns_size);
                        
                        for (int gradient_column = 0; gradient_column < gradient_column_size; gradient_column ++) {

                            int gradient_column_idx = gradient_row_idx + gradient_column;
                            int input_column_idx = input_row_idx + start_gradient_column + gradient_column;

                            sum += gradient_data[gradient_column_idx] * input_data[input_column_idx];
                            
                        }
                    }
                    {
                        auto lock = std::unique_lock<std::mutex>{backward_mutex};
                        kernel.changeNeuron(kernel_column_idx, sum * (kernel._learningRate / active_dimensions)); // breaking here
                        start_gradient_column += stride;
                        kernel_column_idx++;
                    }
                }
                start_gradient_row += stride;
            }
        }
    }
}

void Vision::Convolve(const Tensor &input, Tensor &Output, const Tensor &kernel, Thread_Pool &threadPool, int stride) {

    Output.ResetTensor();
    int activeDimensions = Output.ReturnActiveDimension();

    const auto processor_count = std::thread::hardware_concurrency();
    if (activeDimensions >= processor_count) {

        const int dimensions_per_thread = activeDimensions / processor_count; 
        const int dimensions_per_thread_remainder = activeDimensions % processor_count; 

        for (int i = 0; i < processor_count - 1; i++) { 

            threadPool.enqueue(&Vision::ConvolveInner,
                                    std::ref(input), 
                                    std::ref(Output), 
                                    std::ref(kernel), 
                                    stride, 
                                    i * dimensions_per_thread, 
                                    dimensions_per_thread);
        }
        threadPool.enqueue(&Vision::ConvolveInner,
                                    std::ref(input), 
                                    std::ref(Output), 
                                    std::ref(kernel), 
                                    stride, 
                                    (processor_count-1) * dimensions_per_thread, 
                                    dimensions_per_thread + dimensions_per_thread_remainder);
        
        threadPool.wait();
        
    } else {
        for (int i = 0; i < activeDimensions; i++) {
            Vision::ConvolveInner(input, Output, kernel, stride, i, 1);
        }
    }
}

void Vision::ConvolveInner(const Tensor &input, Tensor &output, const Tensor &kernel, int stride, const int d, const int n_d) {

    const int kernel_row_size = kernel.NumberOfRows();
    const int kernel_column_size = kernel.NumberOfColumns();
    const int kernel_channel_size = kernel.NumberOfChannels();
    const int kernels = kernel.NumberOfDimensions();
    const float *kernel_data = kernel.ReturnData();

    const int input_channel_size = input.NumberOfChannels();
    const int input_row_size = input.NumberOfRows();
    const int input_column_size = input.NumberOfColumns();
    const float *input_data = input.ReturnData();

    const int output_channel_size = output.NumberOfChannels();
    const int output_row_size = output.NumberOfRows();
    const int output_column_size = output.NumberOfColumns();

    // Get mini-batch dimension
    int input_dimension_idx = d * input.NumberOfElementsPerTensor();
    int output_dimension_idx = d * output.NumberOfElementsPerTensor();

    for (int dimension_tracker = 0; dimension_tracker < n_d; dimension_tracker++) {

        // Loop through each kernel
        for (int kernel_dimension = 0; kernel_dimension < kernels; kernel_dimension ++) {

            // Each kernel maps to a channel in the output, so we can get that index
            const int output_channel_idx = output_dimension_idx + (kernel_dimension * output_row_size * output_column_size);
            const int kernel_idx = kernel_dimension * kernel.NumberOfElementsPerTensor();

            // set starting point for kernel row on input
            int start_row = 0;
            int output_row_start = 0;
            while (start_row + kernel_row_size <=  input_row_size) {

                const int output_row_idx = output_channel_idx + (output_row_start * output_column_size);
                int start_column = 0;
                int output_column_start = 0;
                // set starting point for kernel column on input
                while (start_column + kernel_column_size <= input_column_size ) {

                    float sum = 0.0f;
                    const int output_column_idx = output_row_idx + output_column_start;

                    // Loop through each channel in input/kernel
                    for (int input_channel = 0; input_channel < input_channel_size; input_channel++) {

                        const int input_channel_idx = input_dimension_idx + (input_channel * input_row_size * input_column_size);
                        const int kernel_channel_idx = kernel_idx + (input_channel * kernel_row_size * kernel_column_size);

                        // Do Convolution
                        for (int kernel_row = 0; kernel_row < kernel_row_size; kernel_row ++) {

                            const int input_row_idx = input_channel_idx + 
                                                 ((start_row + kernel_row) * input_column_size);

                            const int kernel_row_idx = kernel_channel_idx + (kernel_row * kernel_column_size);

                            for (int kernel_column = 0; kernel_column < kernel_column_size; kernel_column++) {
                            
                                const int kernel_column_idx = kernel_row_idx + kernel_column;
                                const int input_column_idx = input_row_idx + start_column + kernel_column;

                                sum += input_data[input_column_idx] * kernel_data[kernel_column_idx];
                            }
                        }
                    }
                    output.setNeuron(0,output_column_idx, sum);
                    start_column += stride;
                    output_column_start++;
                }
                start_row += stride;
                output_row_start++;
            }
        }
        input_dimension_idx += input.NumberOfElementsPerTensor();
        output_dimension_idx += output.NumberOfElementsPerTensor();
    }
}

// ### Maxpool Methods ###

void Vision::InnerFindMax(const Tensor &input, Tensor &output, int filter_size, int stride, std::vector<unsigned int> &maxpool_indexes, const int d, const int n_d) {

    Dimensions output_dimensions = output.dimensions();

    int input_channels_size = input.NumberOfChannels();
    int input_rows_size = input.NumberOfRows();
    int input_columns_size = input.NumberOfColumns();
    int input_size = input.NumberOfElements();
    const float *input_data = input.ReturnData();

    int output_dimension_index = d * (output_dimensions.channels * output_dimensions.rows * output_dimensions.columns);
    int input_dimension_index = d * (input_channels_size * input_rows_size * input_columns_size);

    for (int dimension_tracker = 0; dimension_tracker < n_d; dimension_tracker++) {

        for (int input_channel = 0; input_channel < input.NumberOfChannels(); input_channel++) {

             int output_channel_idx = output_dimension_index + (input_channel * output_dimensions.rows * output_dimensions.columns);
             int input_channel_idx = input_dimension_index + (input_channel * input_rows_size * input_columns_size);
             
             int start_row_output = 0;
             int start_row = 0 ;

             while (start_row + filter_size <= input_rows_size) {
               
                int start_column = 0;
                int start_column_output = 0;
                int out_row_idx = output_channel_idx + (start_row_output * output.NumberOfColumns());

                while (start_column + filter_size <= input_columns_size) {
               
                    int output_column_idx = out_row_idx + start_column_output;

                    float max_value = std::numeric_limits<float>::min();
                    unsigned int max_index = 0;

                    for (int kernel_row = 0; kernel_row < filter_size; kernel_row++) {

                        int input_row_idx = input_channel_idx + ((kernel_row + start_row) * input_rows_size);

                        for (int kernel_column = 0; kernel_column < filter_size; kernel_column++) {

                            int input_column_idx = input_row_idx + kernel_column + start_column;

                            if (input_data[input_column_idx] >= max_value) {
                                max_value = input_data[input_column_idx];
                                max_index = input_column_idx;
                            }
                        }
                    }
             
                    output.setNeuron(0, output_column_idx, max_value);

                    if (maxpool_indexes.size() > 1) {
                        maxpool_indexes[output_column_idx] = max_index;
                    }
               
                    start_column += stride;
                    start_column_output++;
                }
            
                start_row += stride;
                start_row_output++;
             }
        }
        output_dimension_index += output_dimensions.channels * output_dimensions.rows * output_dimensions.columns;
        input_dimension_index += input_channels_size * input_rows_size * input_columns_size;
    }
}

void Vision::FindMax(const Tensor &input, Tensor &output, Thread_Pool &threadPool, int filter_size, int stride, std::vector<unsigned int> &maxpool_indexes) {

    output.ResetTensor();

    int activeDimensions = output.ReturnActiveDimension();
    const auto processor_count = std::thread::hardware_concurrency();

    if (activeDimensions >= processor_count) {

        
        const int dimensions_per_thread = activeDimensions / processor_count; 
        const int dimensions_per_thread_remainder = activeDimensions % processor_count; 

        for (int i = 0; i < processor_count - 1; i++) { 
            threadPool.enqueue(&Vision::InnerFindMax, 
                                    std::ref(input), 
                                    std::ref(output),
                                    filter_size, 
                                    stride,
                                    std::ref(maxpool_indexes),
                                    i * dimensions_per_thread, 
                                    dimensions_per_thread);
        }
        threadPool.enqueue(&Vision::InnerFindMax, 
                                    std::ref(input), 
                                    std::ref(output),
                                    filter_size, 
                                    stride,
                                    std::ref(maxpool_indexes),
                                    (processor_count-1) * dimensions_per_thread, 
                                    dimensions_per_thread + dimensions_per_thread_remainder);
        
        threadPool.wait();

    } else {

        for (int i = 0; i < activeDimensions; i++) {
            Vision::InnerFindMax(input, output, filter_size, stride, maxpool_indexes, i, 1);
        }

    }
}