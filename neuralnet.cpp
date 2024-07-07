#include "neuralnet.hpp"

Net::Net(vector<int> layers, double l){
    for (int i = 0; i < layers.size(); i++){
        neurons.push_back(vector<double>(layers[i], 0));
        biases.push_back(vector<double>(layers[i], 0));
        propBiases.push_back(vector<double>(layers[i], 0));
    }

    for (int i = 0; i < neurons.size(); i++){
        vector<vector<double>> V;
        vector<double> v(neurons[i-1].size(), 1);
        for (int j = 0; j < neurons[i].size(); j++){
            V.push_back(v);
        }
        weights.push_back(V);
        propWeights.push_back(V);
    }
    
    learnRate = l;
}

void Net::input(vector<double> in){
    for (int i = 0; i < neurons[0].size(); i++){
        neurons[0][i] = in[i];
    }
}

void Net::calculate(){
    for (int i = 1; i < neurons.size(); i++){
        neurons[i] = matrixMult(weights[i], neurons[i-1]);
        for (int j = 0; j < neurons[i].size(); j++){
            neurons[i][j] = sigmoid(neurons[i][j]+biases[i][j]);
        }
    }
}

void Net::backprop(vector<double> out){

    vector<double> errors(neurons.back().size());

    for (int i = 0; i < errors.size(); i++){
        errors[i] = 2*(neurons.back()[i]-out[i])/neurons.back().size();
    }

    vector<double> sigmoids(neurons.back().size());
    for (int i = 0; i < sigmoids.size(); i++){
        errors[i] *= sigmoidThing(neurons.back()[i]);
    }
    


    for (int l = neurons.size()-1; l > 0; l--){
        vector<double> errors2(neurons[l-1].size(), 0);
        for (int r = 0; r < weights[l].size(); r++){
            for (int c = 0; c < weights[l][0].size(); c++){
                propWeights[l][r][c] = neurons[l-1][c] * errors[r];
                propBiases[l][r] = errors[r];
                errors2[c] += weights[l][r][c] * errors[r];
            }
        }
        for (int i = 0; i < neurons[l-1].size(); i++){
            errors2[i] *= sigmoidThing(neurons[l-1][i]);
        }
        errors = errors2;
    }

    for (int l = neurons.size()-1; l > 0; l--){
        for (int r = 0; r < weights[l].size(); r++){
            for (int c = 0; c < weights[l][0].size(); c++){
                weights[l][r][c] -= propWeights[l][r][c] * learnRate;
                biases[l][r] -= propBiases[l][r] * learnRate;

            }
        }
    }
}

void Net::partialbackprop(vector<vector<double>> gradients, vector<double> bgradients){

    vector<double> errors(neurons[neurons.size()-2].size());

    int L = propWeights.size()-1;

    propWeights[L] = gradients;

    propBiases[L] = bgradients;

    for (int r = 0; r < weights.back().size(); r++){
        for (int c = 0; c < weights.back()[0].size(); c++){
            if (neurons[L-1][c] == 0) cout << "pain\n";
            errors[c] += weights[L][r][c] * gradients[r][c] / neurons[L-1][c];
        }
    }
    
    for (int i = 0; i < neurons[L-1].size(); i++){
        errors[i] *= sigmoidThing(neurons[L-1][i]);
    }

    for (int l = neurons.size()-2; l > 0; l--){
        vector<double> errors2(neurons[l-1].size(), 0);
        for (int r = 0; r < weights[l].size(); r++){
            for (int c = 0; c < weights[l][0].size(); c++){
                propWeights[l][r][c] = neurons[l-1][c] * errors[r];
                propBiases[l][r] = errors[r];
                errors2[c] += weights[l][r][c] * errors[r];
            }
        }
        for (int i = 0; i < neurons[l-1].size(); i++){
            errors2[i] *= sigmoidThing(neurons[l-1][i]);
        }
        errors = errors2;
    }

    for (int l = neurons.size()-1; l > 0; l--){
        for (int r = 0; r < weights[l].size(); r++){
            for (int c = 0; c < weights[l][0].size(); c++){
                weights[l][r][c] += propWeights[l][r][c] * learnRate;
                biases[l][r] += propBiases[l][r] * learnRate;

            }
        }
    }
}

void Net::test(vector<vector<double>> gradients, vector<double> out){
    vector<double> errors(neurons[neurons.size()-2].size());

    int L = propWeights.size()-1;

    propWeights[L] = gradients;

    for (int r = 0; r < weights.back().size(); r++){
        for (int c = 0; c < weights.back()[0].size();  c++){
            errors[c] += weights[L][r][c] * gradients[r][c] / neurons[L-1][c] / 10;
        }
    }

    for (int i = 0; i < neurons[L-1].size(); i++){
        errors[i] *= sigmoidThing(neurons[L-1][i]);
    }

    //

    vector<double> errors2(neurons.back().size());

    for (int i = 0; i < errors2.size(); i++){
        errors2[i] = 2*(neurons.back()[i]-out[i])/neurons.back().size();
    }

    vector<double> sigmoids(neurons.back().size());
    for (int i = 0; i < sigmoids.size(); i++){
        errors2[i] *= sigmoidThing(neurons.back()[i]);
    }

    int l = neurons.size()-1;

    vector<double> errors3(neurons[l-1].size(), 0);
    for (int r = 0; r < weights[l].size(); r++){
        for (int c = 0; c < weights[l][0].size(); c++){
            propWeights[l][r][c] = neurons[l-1][c] * errors2[r];
            propBiases[l][r] = errors2[r];
            errors3[c] += weights[l][r][c] * errors2[r];
        }
    }
    for (int i = 0; i < neurons[l-1].size(); i++){
        errors3[i] *= sigmoidThing(neurons[l-1][i]);
    }
    errors2 = errors3;

    for (double f: errors){
        cout << f << "\n";
    }
    cout << "\n";
    for (double f: errors2){
        cout << f << "\n";
    }
}


double Net::totalError(vector<double> e){
    double r = 0.0f;
    for (int i = 0; i < neurons.back().size(); i++){
        r += (neurons.back()[i]-e[i])*(neurons.back()[i]-e[i]);
    }
    r /= neurons.back().size();
    return r;
}

vector<double> Net::matrixMult(vector<vector<double>> a, vector<double> b){
    if (a[0].size() != b.size()){
        return {{0}};
    }

    vector<double> v;


    for (int i = 0; i < a.size(); i++){
        v.push_back(0);
    }


    for (int i = 0; i < a.size(); i++){
        
        for (int k = 0; k < a[0].size(); k++){
            v[i] += a[i][k]*b[k];
        }
    }

    return v;
}

double Net::sigmoid(double d){
    return 1/(1+pow(2.718, -d))-0.6;
}


double Net::sigmoidThing(double d){
    return d*(1-d);
}
