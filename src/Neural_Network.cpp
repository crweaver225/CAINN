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

void Neural_Network::addInputLayer(int *dimensions, int dimension) {
    std::vector<int> input_vector(dimensions, dimensions + dimension);
    std::shared_ptr<Input_layer> input_layer = std::shared_ptr<Input_layer>(new Input_layer(input_vector));
    neural_layers.push_back(input_layer);
}

void Neural_Network::addFullyConnectedLayer(int neurons, int activation_function) {
    std::vector<int> dense_vector(1, neurons);
    std::shared_ptr<Fully_Connected_Layer> full_connected_layer = std::shared_ptr<Fully_Connected_Layer>(new Fully_Connected_Layer(dense_vector, (Activation_Function)activation_function));
    neural_layers.push_back(full_connected_layer);
}

void Neural_Network::addOutputLayer(int neurons, int activation_function) {
    std::vector<int> output_vector(1, neurons);
    std::shared_ptr<Output_Layer> output_layer = std::shared_ptr<Output_Layer>(new Output_Layer(output_vector, (Activation_Function)activation_function));
    neural_layers.push_back(output_layer);
}

void Neural_Network::build() {
    std::cout<<"building neural network..."<<std::endl;
    neural_layers[0]->build(neural_layers[0]);
    for (int i = 1; i < neural_layers.size(); ++i) {
        std::cout<<"building layer number: "<<i<<std::endl;
        neural_layers[i]->build(neural_layers[i-1]);
       // neural_layers[i]->printMetaData();
    }
    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i]->printMetaData();
    }
}

void Neural_Network::execute(float *input) {
    neural_layers[0]->addInput(input);
    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i]->forward_propogate();
    }
    neural_layers.back()->printFinalResults();
}

void Neural_Network::train(float **input, float **targets, int batch_size, int epochs, int input_size) {
    for (int i = 0; i < neural_layers.size(); ++i) {
        neural_layers[i].get()->buildGradient(batch_size);
    }
    std::cout<<"Starting to train neural network..."<<"Number of epochs: "<<epochs<<" Batch size: "<<batch_size<<", input size: "<<input_size<<std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int inputs_for_batch = 0; inputs_for_batch < input_size; inputs_for_batch += batch_size) {

            int final_batch_size = batch_size;
            if (inputs_for_batch + batch_size > input_size) {
                final_batch_size = input_size - inputs_for_batch;
            }

            std::unique_ptr<float*> batch_input = std::unique_ptr<float*>(new float*[batch_size]);
            std::unique_ptr<float*> batch_target = std::unique_ptr<float*>(new float*[batch_size]);

            for (int b = 0; b < final_batch_size; ++b) {
                batch_input.get()[b] = input[inputs_for_batch + b];
                batch_target.get()[b] = targets[inputs_for_batch + b];
            }

            std::cout<<"------- Begin forward propogation ------"<<std::endl;
            neural_layers[0].get()->addInputInBatches(final_batch_size, batch_input.get());
            for (int i = 0; i < neural_layers.size(); ++i) {
                neural_layers[i].get()->forward_propogate();
            }
            std::cout<<"------ End forward propogation ------"<<std::endl;

            std::cout<<"------ Begin backpropogation ------"<<std::endl;
            neural_layers.back().get()->calculateError(batch_target.get(), 0.0f);
            clearGradients();
            backpropogate();
            std::cout<<"------ End backpropogation ------"<<std::endl;
        }
    }
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

float Neural_Network::calculateL2() {

}
