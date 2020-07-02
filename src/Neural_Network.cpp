#include "Neural_Network.h"


Neural_Network::Neural_Network() {
    std::cout<<"Initializing Neural Network"<<std::endl;
}

Neural_Network::~Neural_Network() {
    std::cout<<"Deallocating Neural Network"<<std::endl;
}

Neural_Network::Neural_Network(const Neural_Network &neural_network) {
    std::cout<<"Copy constructor called on Neural Network"<<std::endl;
}

Neural_Network& Neural_Network::operator = (const Neural_Network &neural_network) {
    std::cout<<"Copy assignment operator called on Neural Network"<<std::endl;
}

Neural_Network::Neural_Network(Neural_Network &&neural_network) {
    std::cout<<"Move constructor called on Neural Network"<<std::endl;
}

Neural_Network& Neural_Network::operator=(Neural_Network &&neural_network) {
    std::cout<<"Move assignment operator called on Neural Network"<<std::endl;
}

void Neural_Network::addInputLayer(int dimension) {
    std::vector<int> input_vector{1, 1, dimension};
    std::shared_ptr<Input_layer> input_layer = std::shared_ptr<Input_layer>(new Input_layer(input_vector));
    neural_layers.push_back(input_layer);
}

void Neural_Network::addFullyConnectedLayer(int neurons, int activation_function) {
    std::vector<int> dense_vector{1, 1, neurons};
    std::shared_ptr<Fully_Connected_Layer> full_connected_layer = std::shared_ptr<Fully_Connected_Layer>(new Fully_Connected_Layer(dense_vector, (Activation_Function)activation_function));
    neural_layers.push_back(full_connected_layer);
}

void Neural_Network::addOutputLayer(int neurons, int activation_function) {
    std::vector<int> output_vector{1, 1, neurons};
    std::cout<<activation_function<<std::endl;
    std::shared_ptr<Output_Layer> output_layer = std::shared_ptr<Output_Layer>(new Output_Layer(output_vector, (Activation_Function)activation_function));
    neural_layers.push_back(output_layer);
}

void Neural_Network::setLearningRate(float learning_rate) {
    Tensor::learning_rate = learning_rate;
}

void Neural_Network::set_print_loss_ever_iterations(int iteration) {
    this->print_loss_every_iterations = iteration;
}

void Neural_Network::save_network() {
    Network_Saver network_saver;
    network_saver.save_network(this, this->filePath);
 }

  void Neural_Network::load_network(size_t len, const char* path) {
    std::string path_str(path);
    Network_Saver network_saver;
    network_saver.load_network(this, path_str);
 }

const int Neural_Network::output_dimensions() const {
    return neural_layers.back()->output_dimensions().back();
}

void Neural_Network::set_filepath(const char* path) {
    this->filePath = path;
}

void Neural_Network::save_best_automatically(bool activate) {
    this->save_if_best = activate;
}

void Neural_Network::stop_training_automatically(bool activate) {
    this->stop_automatically = activate;
}

void Neural_Network::build() {
    std::cout<<"building neural network..."<<std::endl;
    
    neural_layers[0]->build(neural_layers[0]);
    for (int i = 1; i < neural_layers.size(); ++i) {
        neural_layers[i]->build(neural_layers[i-1]);
    }
    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i]->printMetaData();
    }
}

const float* Neural_Network::execute(float *input) {
    neural_layers[0]->addInput(input);
    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i]->forward_propogate();
    }
    return neural_layers.back().get()->output_results.get()->returnData();
}

void Neural_Network::train(float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size) {

    this->best_loss = std::numeric_limits<float>::max();
    neural_layers.back().get()->setLossFunction((Loss)loss_function);

    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i].get()->setBatchDimensions(batch_size);
        neural_layers[i].get()->training(true);
    }
    std::cout<<"Starting to train neural network..."<<"Number of epochs: "<<epochs<<" Batch size: "<<batch_size<<", input size: "<<input_size<<std::endl;

    std::unique_ptr<float*> batch_input = std::unique_ptr<float*>(new float*[batch_size]);
    std::unique_ptr<float*> batch_target = std::unique_ptr<float*>(new float*[batch_size]);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int inputs_for_batch = 0; inputs_for_batch < input_size; inputs_for_batch += batch_size) {

            int final_batch_size = batch_size;
            if (inputs_for_batch + batch_size > input_size) {
                final_batch_size = input_size - inputs_for_batch;
            }

            memset(batch_input.get(), 0.0f, batch_size * sizeof(float));
            memset(batch_target.get(), 0.0f, batch_size * sizeof(float));

            for (int b = 0; b < final_batch_size; ++b) {
                batch_input.get()[b] = input[inputs_for_batch + b];
                batch_target.get()[b] = targets[inputs_for_batch + b];
            }

            neural_layers[0].get()->addInputInBatches(final_batch_size, batch_input.get());

            for (int i = 0; i < neural_layers.size(); ++i) {
                neural_layers[i].get()->setActiveDimensions(final_batch_size);
                neural_layers[i].get()->forward_propogate();
            }

            neural_layers.back().get()->calculateError(batch_target.get(), calculateL2());
        
            clearGradients();
            backpropogate();
        }

        if (epoch % this->print_loss_every_iterations == 0) {
            std::cout<<" iteration: "<<epoch<<" ";
            neural_layers.back().get()->printError();
        }

        float loss = neural_layers.back().get()->returnLoss();

        if (save_if_best) {
            if (loss < best_loss) {
                save_network();
                best_loss = loss;
            }
        }

        if (stop_automatically) {
            if (loss == 0) {
                std::cout<<"stopping training early, loss has reached zero"<<std::endl;
                break;
            }
        }
        neural_layers.back().get()->resetLoss();
    }

    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i].get()->setBatchDimensions(1);
        neural_layers[i].get()->training(false);
    }
    std::cout<<"Training complete"<<std::endl;
}

void Neural_Network::backpropogate() {
    for (int i = neural_layers.size() - 1; i > 0; --i) {
        neural_layers[i].get()->backpropogate();
    }
}

void Neural_Network::clearGradients() {
    for (int i = neural_layers.size() - 1; i > 0; --i) {
        neural_layers[i].get()->clearGradient();
    }
}    

const float Neural_Network::calculateL2() const {
    float l2_value = 0.0f;
    for (int i = neural_layers.size() - 1; i > 0; i--) {
        l2_value += neural_layers[i]->returnL2();
    }
    return l2_value;
}
