#include "Neural_Network.h"

void Neural_Network::AddInputLayer(int *dimension, int size) {

   std::vector<int> input_vector{1,1,1,1};
    if (size == 1) {
        input_vector = {1,1,1,dimension[0]};
    } else if (size == 2) {
        input_vector = {1,1,dimension[0], dimension[1]};
    } else if (size == 3) {
        input_vector = {1,dimension[0], dimension[1], dimension[2]};
    }
    std::shared_ptr<Input_layer> input_layer = std::shared_ptr<Input_layer>(new Input_layer(input_vector));
    _neuralLayers.push_back(input_layer);
}

void Neural_Network::AddFullyConnectedLayer(int neurons, int activation_function) {
    std::vector<int> dense_vector{1,1,1,neurons};
    std::shared_ptr<Fully_Connected_Layer> full_connected_layer = std::shared_ptr<Fully_Connected_Layer>(new Fully_Connected_Layer(dense_vector, (Activation_Function)activation_function));
    _neuralLayers.push_back(full_connected_layer);
}

void Neural_Network::AddMaxpoolLayer(int kernals, int stride) {
    std::vector<int> maxpool_vector{1,1,kernals,stride};
    std::shared_ptr<Maxpool_Layer> maxpool_layer = std::shared_ptr<Maxpool_Layer>(new Maxpool_Layer(maxpool_vector));
    _neuralLayers.push_back(maxpool_layer);
}

void Neural_Network::AddDropoutLayer(float dropped) {
    std::vector<int> dropout_vector{1,1,1,1};
    std::shared_ptr<Dropout_Layer> dropout_layer = std::shared_ptr<Dropout_Layer>(new Dropout_Layer(dropout_vector, dropped));
    _neuralLayers.push_back(dropout_layer);
    _droppoutLayerExists = true;
}

void Neural_Network::AddFlattenLayer() {
    std::shared_ptr<Flatten_Layer> flatten_layer = std::shared_ptr<Flatten_Layer>(new Flatten_Layer());
    _neuralLayers.push_back(flatten_layer);
}

void Neural_Network::AddEmbeddingLayer(int unique_words_length, int output) {
    std::vector<int> embedding_vector{1, 1, unique_words_length, output};
    std::shared_ptr<Embedding_Layer> embedding_layer = std::shared_ptr<Embedding_Layer>(new Embedding_Layer(embedding_vector));
    _neuralLayers.push_back(embedding_layer);
}

void Neural_Network::AddOutputLayer(int neurons, int activation_function) {
    std::vector<int> output_vector{1, 1, 1, neurons};
    std::shared_ptr<Output_Layer> output_layer = std::shared_ptr<Output_Layer>(new Output_Layer(output_vector, (Activation_Function)activation_function));
    _neuralLayers.push_back(output_layer);
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

void Neural_Network::LoadNetwork(size_t len, const char* path) {
    std::string path_str(path);
    Network_Saver network_saver;
    network_saver.LoadNetwork(this, path_str);
}

const int Neural_Network::OutputDimensions() const {
    return _neuralLayers.back()->OutputDimensions().back();
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
    _neuralLayers[0]->Build(_neuralLayers[0]);
    for (int i = 1; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i]->Build(_neuralLayers[i-1]);
    }
    for (int i = 0; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i]->PrintMetaData();
    }
}

const float* Neural_Network::Execute(float *input) {
    _neuralLayers[0]->AddInput(input);
    for (int i = 0; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i]->ForwardPropogate();
    }
    return _neuralLayers.back().get()->_outputResults.get()->ReturnData();
}

void Neural_Network::Train(float **input, float **targets, int batch_size, int epochs, int loss_function, int input_size) {

    this->_bestLoss = std::numeric_limits<float>::max();

    _neuralLayers.back().get()->SetLossFunction((Loss)loss_function);

    for (int i = 0; i < _neuralLayers.size(); ++i) {
        _neuralLayers[i].get()->SetBatchDimensions(batch_size);
        _neuralLayers[i].get()->Training(true);
    }

    std::cout<<"Starting to train neural network..."<<"Number of epochs: "<<epochs<<" Batch size: "<<batch_size<<", input size: "<<input_size<<std::endl;

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

            _neuralLayers[0].get()->AddInputInBatches(final_batch_size, batch_input.get());

            for (int i = 0; i < _neuralLayers.size(); ++i) {
                _neuralLayers[i].get()->SetActiveDimensions(final_batch_size);
                _neuralLayers[i].get()->ForwardPropogate();
            }
            _neuralLayers.back().get()->CalculateError(batch_target.get(), CalculateL2());
            ClearGradients();
            Backpropogate();
        }

        if (epoch % this->_printLossEveryIterations == 0) {
            std::cout<<" iteration: "<<epoch + 1<<" ";
            _neuralLayers.back().get()->PrintError();
        }

        float loss = _neuralLayers.back().get()->ReturnLoss();

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

        _neuralLayers.back().get()->ResetLoss();
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
    std::random_shuffle ( myvector.begin(), myvector.end() );
    for (int i = 0; i < input_size - 2; i++) {
        std::swap(input[myvector[i]], input[myvector[i + 1]]);
        std::swap(targets[myvector[i]], targets[myvector[i + 1]]);
    }
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
    for (int i = _neuralLayers.size() - 1; i > 0; --i) {
        _neuralLayers[i].get()->Backpropogate();
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
