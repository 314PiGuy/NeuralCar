#include <iostream>
#include <array>
#include <cmath>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

class Car
{
private:
    double angleInc = 3.141 / 3.0f;
public:
    double heading;
    array<double, 2> center;
    array<array<double, 2>, 4> corners;
    double size;
    Car(double s, double h, array<double, 2> c);
    void move(double d);
    void rotate(double angle);
    vector<double> getDists(Image im);
    array<array<double, 2>, 3> drawRay(Image im);
    ~Car();
};


