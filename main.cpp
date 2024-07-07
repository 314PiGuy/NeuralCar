#include <SFML/Graphics.hpp>
#include <random>
#include "Car.hpp"
#include "neuralnet.hpp"

using namespace std;
using namespace sf;


RenderWindow window(VideoMode(800, 800), "N00000000");


vector<vector<float>> gradients;

template <typename S>
ostream& operator<<(ostream& os, const vector<S>& vector)
{
    for (auto element : vector) {
        os << element << " ";
    }
    return os;
}

ConvexShape makeCar(Car car){
   ConvexShape c;
   c.setPointCount(4);
   for (int i = 0; i < 4; i++){
    c.setPoint(i, Vector2f(car.corners[i][0], car.corners[i][1]));
   }
   return c;
}

void randomize(Net &net){
    random_device rd;
    std::mt19937 engine(rd());
    normal_distribution<> d(-1, 1);
    for (int l = 0; l < net.neurons.size(); l++){
        // int in = net.weights[l][0].size();
        // int out = net.weights[l].size();
        // float dist = sqrt(6/(in+out));
        // normal_distribution<> d(-dist, dist);
        for (int r = 0; r < net.weights[l].size(); r++){
            for (int c = 0; c < net.weights[l][0].size(); c++){
                net.weights[l][r][c] = d(engine);
            }
        }
        for (int r = 0; r < net.biases[l].size(); r++){
            net.biases[l][r] = d(engine);
        }
    }
}

void train(Net &net, vector<float> in, vector<float> out){
    net.input(in);
    net.calculate();
    net.backprop(out);
}


float getAngle(vector<float> dists, Net &net){
    net.input(dists);
    net.calculate();
    float f = net.neurons.back()[0];
    return f*3.14159;
}

bool crashed(Car car, Image im){
    for (int i = 1; i <= 2; i++){
        auto c = car.corners[i];
        if (im.getPixel(c[0], c[1]) == Color::Black){
            return true;
        }
        if (c[0] < 0 || c[1] < 0 || c[0] > im.getSize().x || c[1] > im.getSize().y){
            return true;
        }
    }

    return false;
}

int runCar(Car car, Net &net, Image map){
    int n = 1;
    for (int i = 0; i < 200; i++){
        car.rotate(car.heading+getAngle(car.getDists(map), net));
        car.move(2);
        if (crashed(car, map)) break;
        n++;
    }
    return n;
}

int runCar2(Car car, Net &net, Image map, RectangleShape rect){
    int n = 1;
    for (int i = 0; i < 200; i++){
        car.rotate(car.heading+getAngle(car.getDists(map), net));
        car.move(0.2);
        window.clear();
        window.draw(rect);
        ConvexShape carShape = makeCar(car);
        carShape.setFillColor(Color::Red);
        for (int i = 0; i < 4; i++){
            window.draw(carShape);
        }
        window.display();
        if (crashed(car, map)) break;
        n++;
    }
    return n;
}

void trainCar(Car car, Net &net, Image map){

    int l = net.weights.size()-1;
    vector<vector<float>> g = gradients;
    for (int r = 0; r < net.weights[l].size(); r++){
        for (int c = 0; c < net.weights[l][0].size(); c++){
            net.weights[l][r][c] += 0.1;
            int moves = runCar(car, net, map);
            net.weights[l][r][c] -= 0.1;
            int moves2 = runCar(car, net, map);
            g[r][c] = 10*(moves-moves2);
        }
    }
    vector<float> v;
    for (int n = 0; n < net.biases[l].size(); n++){
        net.biases[l][n] += 0.1;
        int moves = runCar(car, net, map);
        net.biases[l][n] -= 0.1;
        int moves2 = runCar(car, net, map);
        v.push_back(10*(moves-moves2));
    }
    net.partialbackprop(g, v);
}

void trainCar2(Car car, Net &net, Image map){
    for (int l = 0; l < net.weights.size(); l++){
        for (int r = 0; r < net.weights[l].size(); r++){
            for (int c = 0; c < net.weights[l][0].size(); c++){
                net.weights[l][r][c] += 0.1;
                int moves = runCar(car, net, map);
                net.weights[l][r][c] -= 0.1;
                int moves2 = runCar(car, net, map);
                net.weights[l][r][c] += 10*(moves-moves2);
            }
            net.biases[l][r] += 0.1;
            int moves = runCar(car, net, map);
            net.biases[l][r] -= 0.1;
            int moves2 = runCar(car, net, map);
            net.biases[l][r] += 10*(moves-moves2);
        }
    }

}




int main()
{

    vector<int> l = {3, 4, 4, 1};
    Net net = Net(l, 0.1);
    randomize(net);

    Image mapImage;
    mapImage.loadFromFile("Assets/path2.png");

    Car c = Car(10, 0, {400, 70});  


    vector<float> v;

    for (int i = 0; i < net.neurons[net.neurons.size()-2].size(); i++){
        v.push_back(0.0f);
    }

    for (int i = 0; i < net.neurons.back().size(); i++){
        gradients.push_back(v);
    }


    Texture t;
    t.loadFromFile("Assets/path2.png");

    RectangleShape rect(Vector2f(800, 800));
    rect.setTexture(&t);

    runCar2(c, net, mapImage, rect);

    cout << runCar(c, net, mapImage) << "\n";
    for (int i = 0; i < 100; i++){
        // if (i%10 == 0) runCar2(c, net, mapImage);
        trainCar(c, net, mapImage);
    }


    for (auto l: net.weights){
        for (auto w: l){
            for (auto f: w)
                cout << f << "\n";
        }
    }

    cout << endl;

    for (auto l: net.biases){
        for (auto w: l){
            cout << w << endl;
        }
    }

    cout << endl;

    cout << runCar(c, net, mapImage) << "\n\n";

    // for (auto d: c.getDists(mapImage)){
    //     cout << d << "\n";
    // }



    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed){
                window.close();
            }
        }

        window.clear();

        window.draw(rect);

        if (!crashed(c, mapImage)){
            c.move(2);
            // cout << c.getDists(mapImage) << endl;
            c.rotate(c.heading+getAngle(c.getDists(mapImage), net));
        }

        ConvexShape carShape = makeCar(c);
        carShape.setFillColor(Color::Red);
        for (int i = 0; i < 4; i++){
            window.draw(carShape);
        }

        array<array<double, 2>, 3> rays = c.drawRay(mapImage);

        for (auto p: c.drawRay(mapImage)){
            Vertex line[] = {Vertex(Vector2f(c.center[0], c.center[1])), Vertex(Vector2f(p[0], p[1]))};
            Vertex v;
            line[0].color = Color::Blue; line[1].color = Color::Blue;
            window.draw(line, 2, Lines);
        }


        window.display();


    }
    return 0;
}