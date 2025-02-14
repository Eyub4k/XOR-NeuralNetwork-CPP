#include <vector>   
#include <iostream>  
#include <cmath>     
#include <cstdlib> 

using namespace std;

class Neuron
{
public:
    vector<double> weights; // poids
    double bias; // biais
    double output;
    double delta; // erreur local

    Neuron(int input_size) 
    {
        for (int i = 0; i < input_size; i++) // init des poids aleatoirements
        {
            weights.push_back((double)rand() / RAND_MAX * 2.0 - 1.0); // valeurs entre -1 et 1
        }
        bias = (double)rand() / RAND_MAX * 2.0 - 1.0;   // init des biais
        delta = 0.0;
    }

    double activate(double x)  // fonction d'activation
    {
        return tanh(x); 
        //return 1.0 / (1.0 + exp(-x)); // version sigmoide mais pas addapter pour ce jdd car ce n'est pas centree vers le 0 
    }

    double activate_derivative(double x) 
    {
        double tanh_x = tanh(x);
        return 1.0 - tanh_x * tanh_x;  // dérivée de tanh
        //return x * (1 - x);  dérivée de la sigmoide
    }
    
    // & permet d'optimiser la performance en memoire car il le met en reference au lieu de creer une copie  
    double forward(const vector<double>& inputs)   
    {
        double sum = bias;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            sum += inputs[i] * weights[i];
        }
        output = activate(sum);
        return output;
    }
};

class Layer 
{
public:
    vector<Neuron> neurons;
    vector<double> inputs;  // stockage des entrées pour la backpropagation

    Layer(int num_neurons, int input_size) 
    {
        for (int i = 0; i < num_neurons; i++) 
        {
            neurons.push_back(Neuron(input_size));
        }
    }

    vector<double> forward(const vector<double>& layer_inputs) 
    {
        inputs = layer_inputs;  // stockage pour la backpropagation
        vector<double> outputs;
        for (Neuron& neuron : neurons) 
        {
            outputs.push_back(neuron.forward(layer_inputs));
        }
        return outputs;
    }
};

class NeuralNetwork
{
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<int>& architecture)
    {
        for (size_t i = 1; i < architecture.size(); i++) 
        {
            layers.push_back(Layer(architecture[i], architecture[i - 1]));  
            // architecture[i] = nb de neurones de la couche actuelle
            // architecture[i-1] = nb de neurones de la couche précédente (donc nb d'entrées pour cette couche)
        }
    }

    vector<double> forward(const vector<double>& inputs)
    {
        vector<double> activations = inputs;
        for (Layer& layer : layers) 
        {
            activations = layer.forward(activations);
        }
        return activations;
    }

    void backpropagate(const vector<double>& inputs, const vector<double>& targets, double learning_rate) 
    {
        // passe avant
        vector<double> current_inputs = inputs;
        vector<vector<double>> all_outputs;
        all_outputs.push_back(inputs);
        
        for (Layer& layer : layers) 
        {
            current_inputs = layer.forward(current_inputs);
            all_outputs.push_back(current_inputs);
        }

        // backpropagation
        for (int i = layers.size() - 1; i >= 0; i--) 
        {
            Layer& layer = layers[i];
            
            if (i == layers.size() - 1) 
            {
                // Couche de sortie
                for (size_t j = 0; j < layer.neurons.size(); j++) 
                {
                    double error = targets[j] - layer.neurons[j].output;
                    layer.neurons[j].delta = error * layer.neurons[j].activate_derivative(layer.neurons[j].output);
                }
            } 
            else 
            {
                // couches cachees
                Layer& next_layer = layers[i + 1];
                for (size_t j = 0; j < layer.neurons.size(); j++) 
                {
                    double error = 0.0;
                    for (size_t k = 0; k < next_layer.neurons.size(); k++) 
                    {
                        error += next_layer.neurons[k].delta * next_layer.neurons[k].weights[j];
                    }
                    layer.neurons[j].delta = error * layer.neurons[j].activate_derivative(layer.neurons[j].output);
                }
            }

            // maj des poids et des biais
            vector<double>& layer_inputs = all_outputs[i];
            for (Neuron& neuron : layer.neurons) 
            {
                for (size_t j = 0; j < neuron.weights.size(); j++) 
                {
                    neuron.weights[j] += learning_rate * neuron.delta * layer_inputs[j];
                }
                neuron.bias += learning_rate * neuron.delta;
            }
        }
    }

    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs, double learning_rate) 
    {
        for (int epoch = 0; epoch < epochs; epoch++) 
        {
            double total_error = 0.0;
            for (size_t i = 0; i < inputs.size(); i++) 
            {
                vector<double> prediction = forward(inputs[i]);
                backpropagate(inputs[i], targets[i], learning_rate);
                
                // calcul de l'rreur
                for (size_t j = 0; j < targets[i].size(); j++) 
                {
                    total_error += pow(targets[i][j] - prediction[j], 2);
                }
            }
            
            if (epoch % 1000 == 0) 
            {
                cout << "Époque " << epoch << ", Erreur: " << total_error / inputs.size() << endl;
            }
        }
    }
};

int main() {
    srand(time(0));
    vector<int> architecture = {2, 4, 1};  // 2 ent, 4 neurones caches, 1 sortie
    NeuralNetwork net(architecture);

    // jeu de donnee XOR
    vector<vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    // label 
    vector<vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // entrainement
    net.train(inputs, targets, 10000, 0.1);  // 10000 époques, taux d'apprentissage de 0.1

    cout << "\nTest du réseau après l'entraînement:" << endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        vector<double> output = net.forward(inputs[i]);
        cout << "Entrée: " << inputs[i][0] << " " << inputs[i][1];
        cout << " - Sortie: " << output[0];
        cout << " - Attendu: " << targets[i][0] << endl;
    }

    return 0;
}