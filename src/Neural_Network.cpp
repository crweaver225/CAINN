#include "Neural_Network.h"

void Neural_Network::AddInputLayer(int *dimension, int size) {
   Dimensions dimensions;
   if (size == 1) {
        dimensions = {1,1,1,dimension[0]};
    } else if (size == 2) {
        dimensions = {1,1,dimension[0], dimension[1]};
    } else if (size == 3) {
        dimensions = {1,dimension[0], dimension[1], dimension[2]};
    } else if (size == 4) {
        dimensions = {dimension[0],dimension[1], dimension[2], dimension[3]};
    }
    _neuralLayers.push_back(std::make_unique<Input_layer>(Input_layer(dimensions)));
}

void Neural_Network::AddFullyConnectedLayer(int neurons, int activation_function) {
    _neuralLayers.push_back(std::make_unique<Fully_Connected_Layer>(Fully_Connected_Layer(Dimensions{1,1,1,neurons}, 
        (Activation_Function)activation_function)));
}

void Neural_Network::AddDropoutLayer(float dropped) {
    _neuralLayers.push_back(std::make_unique<Dropout_Layer>(Dropout_Layer({1,1,1,1}, dropped)));
    _droppoutLayerExists = true;
}

void Neural_Network::AddFlattenLayer() {
    _neuralLayers.push_back(std::make_unique<Flatten_Layer>(Flatten_Layer()));
}

void Neural_Network::AddEmbeddingLayer(int unique_words_length, int output) {
    _neuralLayers.push_back(std::make_unique<Embedding_Layer>(Embedding_Layer(Dimensions{1, 1, unique_words_length, output})));
}

void Neural_Network::AddOutputLayer(int neurons, int activation_function) {
    _neuralLayers.push_back(std::make_unique<Output_Layer>(Output_Layer(Dimensions{1, 1, 1, neurons}, 
        (Activation_Function)activation_function)));
}

void Neural_Network::SetLearningRate(float learning_rate) {
    Tensor::_learningRate = learning_rate;
}

void Neural_Network::SetPrintLossEverIterations(int iteration) {
    this->_printLossEveryIterations = iteration;
}

void Neural_Network::SaveNetwork() {
    std::cout<<"saving network..."<<std::endl;
    Network_Saver network_saver;
    network_saver.SaveNetwork(this, this->_filePath);
 }

 void Neural_Network::TurnOnRegularization(bool activate) {
    _applyL2Regularization = activate;
 }

void Neural_Network::LoadNetwork(size_t len, const char* path) {
    std::string path_str(path);
    Network_Saver network_saver;
    network_saver.LoadNetwork(this, path_str);
}

const int Neural_Network::OutputDimensions() const {
    return _neuralLayers.back()->ReturnDimensions().columns;
}

void Neural_Network::SetFilepath(const char* path) {
    this->_filePath = path;
}

void Neural_Network::SaveBestAutomatically(bool activate) {
    this->_saveIfBest = activate;
}

void Neural_Network::SetShuffleDataFlag(bool activate) {
    this->_shuffleDataPerEpoch = activate;
}

void Neural_Network::StopTrainingAutomatically(bool activate) {
    this->_stopAutomatically = activate;
}

void Neural_Network::Build() {

    _input_layer = dynamic_cast<Input_layer*>(_neuralLayers[0].get());
    if (_input_layer != nullptr) {

        this->_output_layer = dynamic_cast<Output_Layer*>(_neuralLayers.back().get());
        if (_output_layer != nullptr) {

            _input_layer->Build(_neuralLayers[0].get());

            for (int i = 1; i < _neuralLayers.size(); ++i) {
                 _neuralLayers[i]->Build(_neuralLayers[i-1].get());
            }

            for (int i = 0; i < _neuralLayers.size(); ++i) {
                _neuralLayers[i]->PrintMetaData();
            }

        } else {
            std::cout<<"ERROR ~ All Neural Networks must end with an output layer"<<std::endl;
        }
    } else {
        std::cout<<"ERROR ~ All Neural Networks must start with an input layer"<<std::endl;
    }
}

const float* Neural_Network::Execute(float *input) {
    Tensor const* output = _input_layer->AddInput(input);
    for (int i = 0; i < _neuralLayers.size(); ++i) {
        output = _neuralLayers[i]->ForwardPropogate(output);
    }
    return output->ReturnData();
}

void Neural_Network::Train(float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size) {

    _bestLoss = std::numeric_limits<float>::max();

   _output_layer->SetLossFunction((Loss)loss_function);

    for (int i = 0; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i].get()->SetBatchDimensions(batch_size);
        _neuralLayers[i].get()->Training(true);
    }

    std::cout<<"Starting to train neural network..."
            <<"Number of epochs: "
            <<epochs
            <<" Batch size: "
            <<batch_size
            <<", input size: "
            <<input_size
            <<std::endl;

    std::unique_ptr<float*> batch_input = std::unique_ptr<float*>(new float*[batch_size]);
    std::unique_ptr<float*> batch_target = std::unique_ptr<float*>(new float*[batch_size]);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {

        RandomizeDropout();

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

            Tensor const* output = _input_layer->AddInputInBatches(final_batch_size, batch_input.get());

            for (int i = 1; i < _neuralLayers.size(); ++i) {
                _neuralLayers[i]->SetActiveDimensions(final_batch_size);
                output = _neuralLayers[i]->ForwardPropogate(output);
            }

            _output_layer->CalculateError(batch_target.get(), _applyL2Regularization ? CalculateL2() : 0.0f);
            ClearGradients();
            Backpropogate();
        }

        if (epoch % this->_printLossEveryIterations == 0) {
            std::cout<<" iteration: "<<epoch + 1<<" ";
            _output_layer->PrintError();
        }

        float loss = _output_layer->ReturnLoss();

        if (_saveIfBest) {
            if (loss < _bestLoss) {
                SaveNetwork();
                _bestLoss = loss;
            }
        }

        if (_stopAutomatically) {
            if (loss == 0) {
                std::cout<<"stopping training early, loss has reached zero"<<std::endl;
                break;
            }
        }

        if (_shuffleDataPerEpoch) {
            ShuffleTrainingData(input, targets, input_size);
        }

       _output_layer->ResetLoss();
    }
    
    for (int i = 0; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i].get()->SetBatchDimensions(1);
        _neuralLayers[i].get()->Training(false);
    }
    
    std::cout<<"Training complete"<<std::endl;
}

void Neural_Network::ShuffleTrainingData(float **input, float **targets, int input_size) {
    std::cout<<"Shuffling data ..."<<std::endl;

    std::vector<int> myvector;
    for (int i=0; i<input_size; ++i) myvector.push_back(i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle ( myvector.begin(), myvector.end(), g);

    for (int i = 0; i < input_size - 2; i++) {
        std::swap(input[myvector[i]], input[myvector[i + 1]]);
        std::swap(targets[myvector[i]], targets[myvector[i + 1]]);
    }
}

std::vector<Neural_Layer *> Neural_Network::network() const {
    std::vector<Neural_Layer *> network_layers{};
    for (int i = 0; i < _neuralLayers.size(); ++i) {
        network_layers.push_back(_neuralLayers[i].get());
    }
    return network_layers;
}

void Neural_Network::RandomizeDropout() {
    if (_droppoutLayerExists) {
        for (int i = _neuralLayers.size() - 1; i > 0; --i) {
            Dropout_Layer *dl = dynamic_cast<Dropout_Layer*>(_neuralLayers[i].get());
            if (dl) {
                dl->randomizeDropped();
            }
        }
    }
}

void Neural_Network::Backpropogate() {
    Tensor *gradient = _output_layer->ReturnError();
    for (int i = _neuralLayers.size() - 1; i > 0; --i) {
        gradient = _neuralLayers[i].get()->Backpropogate(gradient);
    }
}

void Neural_Network::ClearGradients() {
    for (int i = _neuralLayers.size() - 1; i > 0; --i) {
        _neuralLayers[i].get()->ClearGradient();
    }
}    

const float Neural_Network::CalculateL2() const {
    float l2_value = 0.0f;
    for (int i = _neuralLayers.size() - 1; i > 0; i--) {
        l2_value += _neuralLayers[i]->ReturnL2();
    }
    return l2_value;
}