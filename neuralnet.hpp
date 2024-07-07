#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class Net
{
private:
    
public:
    vector<vector<double>> neurons;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    vector<vector<vector<double>>> propWeights;
    vector<vector<double>> propBiases;
    double learnRate;
    Net(vector<int> layers, double f);
    void test(vector<vector<double>> v, vector<double> out);
    vector<double> matrixMult(vector<vector<double>> a, vector<double> b);
    double sigmoidThing(double d);
    double sigmoid(double d);
    void calculate();
    void input(vector<double> in);
    void backprop(vector<double> out);
    void partialbackprop(vector<vector<double>> out, vector<double> b);
    double totalError(vector<double> e);
};