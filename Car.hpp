#include <iostream>
#include <array>
#include <cmath>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

class Car
{
private:
    float angleInc = 3.141 / 3.0f;
public:
    float heading;
    array<float, 2> center;
    array<array<float, 2>, 4> corners;
    float size;
    Car(float s, float h, array<float, 2> c);
    void move(float d);
    void rotate(float angle);
    vector<float> getDists(Image im);
    array<array<double, 2>, 3> drawRay(Image im);
    ~Car();
};


