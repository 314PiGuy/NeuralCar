#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class Net
{
private:
    
public:
    vector<vector<float>> neurons;
    vector<vector<vector<float>>> weights;
    vector<vector<float>> biases;
    vector<vector<vector<float>>> propWeights;
    vector<vector<float>> propBiases;
    float learnRate;
    Net(vector<int> layers, float f);
    void test(vector<vector<float>> v, vector<float> out);
    vector<float> matrixMult(vector<vector<float>> a, vector<float> b);
    float sigmoidThing(float d);
    float sigmoid(float d);
    void calculate();
    void input(vector<float> in);
    void backprop(vector<float> out);
    void partialbackprop(vector<vector<float>> out, vector<float> b);
    float totalError(vector<float> e);
};