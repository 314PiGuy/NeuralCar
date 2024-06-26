#include <SFML/Graphics.hpp>
#include <random>
#include "Car.hpp"
#include "neuralnet.hpp"

using namespace std;
using namespace sf;

RenderWindow window(VideoMode(800, 800), "N00000000");

vector<vector<float>> gradients;

ConvexShape makeCar(Car car){
   ConvexShape c;
   c.setPointCount(4);
   for (int i = 0; i < 4; i++){
    c.setPoint(i, Vector2f(car.corners[i][0], car.corners[i][1]));
   }
//    r[0][0] = Vertex(Vector2f(car.corners[0][0], car.corners[0][1])); r[0][1] =  Vertex(Vector2f(car.corners[1][0], car.corners[1][1]));
//    r[1][0] = Vertex(Vector2f(car.corners[1][0], car.corners[1][1])); r[1][1] =  Vertex(Vector2f(car.corners[2][0], car.corners[2][1]));
//    r[2][0] = Vertex(Vector2f(car.corners[2][0], car.corners[2][1])); r[2][1] =  Vertex(Vector2f(car.corners[3][0], car.corners[3][1]));
//    r[3][0] = Vertex(Vector2f(car.corners[3][0], car.corners[3][1])); r[3][1] =  Vertex(Vector2f(car.corners[0][0], car.corners[0][1]));
   return c;
}

void randomize(Net &net){
    random_device rd;
    std::mt19937 engine(rd());
    normal_distribution<> d(-1, 1);
    for (int l = 0; l < net.neurons.size(); l++){
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


float getAngle(vector<float> dists, Net net){
    net.input(dists);
    net.calculate();
    float f = net.neurons.back()[0];
    return f*3.14159;
}

bool crashed(Car car, Image im){
    for (array<float, 2> c: car.corners){
        if (im.getPixel(c[0], c[1]) == Color::Black){
            return true;
        }
    }

    return false;
}

int runCar(Car car, Net net, Image map){
    for (int i = 0; i < 200; i++){
        car.rotate(car.heading+getAngle(car.getDists(map), net));
        car.move(2);
        if (crashed(car, map)) return i;
        i++;
    }
    return 199;
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
    net.partialbackprop(g);

}




int main()
{

    vector<int> l = {3, 4, 4, 1};
    Net net = Net(l, 0.1);
    randomize(net);

    Image mapImage;
    mapImage.loadFromFile("Assets/path1.png");

    Car c = Car(20, 0); 

    vector<float> v;

    for (int i = 0; i < net.neurons[net.neurons.size()-2].size(); i++){
        v.push_back(0.0f);
    }

    for (int i = 0; i < net.neurons.back().size(); i++){
        gradients.push_back(v);
    }

    Texture t;
    t.loadFromFile("Assets/path1.png");


    RectangleShape rect(Vector2f(800, 800));
    rect.setTexture(&t);

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


        Vertex lines[2];

        ConvexShape carShape = makeCar(c);
        carShape.setFillColor(Color::Red);
        for (int i = 0; i < 4; i++){
            window.draw(carShape);
        }


        window.display();


    }
    return 0;
}


// int main(){

    // vector<int> l = {2, 3, 4, 2};
    // Net net = Net(l, 0.1);

    // net = randomize(net);

    // net.input({0.5, 0.5});
    // net.calculate();

    // vector<float> v;

    // vector<vector<float>> gradients;


    // for (int i = 0; i < net.neurons[net.neurons.size()-2].size(); i++){
    //     v.push_back(0.0f);
    // }

    // for (int i = 0; i < net.neurons.back().size(); i++){
    //     gradients.push_back(v);
    // }

    
    // for (int r = 0; r < net.weights.back().size(); r++){
    //     for (int c = 0; c < net.weights.back()[0].size(); c++){
    //         net.weights.back()[r][c] += 0.1;
    //         net.input({0.5, 0.5});
    //         net.calculate();
    //         float e1 = net.totalError({0, 1});
    //         net.weights.back()[r][c] -= 0.1;
    //         net.input({0.5, 0.5});
    //         net.calculate();
    //         float e2 = net.totalError({0, 1});
    //         gradients[r][c] = 10*(e1-e2);
    //     }
    // }

    // // net.test(gradients, {0, 1});
    
    
    // for (int i = 0; i < net.weights.size(); i++){
    //     for (int j = 0; j < net.weights[i].size(); j++){
    //         for (int k = 0; k < net.weights[i][j].size(); k++){
    //             net.weights[i][j][k] += 0.1;
    //             net.input({0.5, 0.5});
    //             net.calculate();
    //             float e1 = net.totalError({0, 1});
    //             net.weights[i][j][k] -= 0.1;
    //             net.input({0.5, 0.5});
    //             net.calculate();
    //             float e2 = net.totalError({0, 1});

    //             cout << 10*(e1-e2) << "\n";
    //         }
    //     }
    // }

    // cout << "\n";


    // net.input({0.5, 0.5});
    // net.calculate();
    // net.partialbackprop(gradients);

    // for (int i = 0; i < net.weights.size(); i++){
    //     for (int j = 0; j < net.weights[i].size(); j++){
    //         for (int k = 0; k < net.weights[i][j].size(); k++){
    //             cout << net.propWeights[i][j][k] << "\n";
    //         }
    //     }
    // }

    // cout << "\n";

    // net.input({0.5, 0.5});
    // net.calculate();
    // net.backprop({0, 1});

    // for (int i = 0; i < net.weights.size(); i++){
    //     for (int j = 0; j < net.weights[i].size(); j++){
    //         for (int k = 0; k < net.weights[i][j].size(); k++){
    //             cout << net.weights[i][j][k] << "\n";
    //         }
    //     }
    // }
    // cout << "\n";

    // return 0;



    // for (int i = 0; i < 100000; i++){
    //     for (int i = 0; i <= 1; i++){
    //         for (int j = 0; j <= 1; j++){
    //             net.input({i/1.0f, j/1.0f});
    //             net.calculate();
    //             net.backprop({((int)(i!=j))/1.0f, 1-((int)(i!=j))/1.0f});
    //         }
    //     }
    // }

    // for (int i = 0; i < net.layers.size(); i++){
    //     for (int j = 0; j < net.layers[i].weights.size(); j++){
    //         // for (int k = 0; k < net.layers[i].weights[0].size(); k++){
    //         //     cout << net.layers[i].weights[j][k] << "\n";
    //         // }
    //         cout << net.layers[i].biases[j] << "\n";
    //     }
    // }
    // cout << "\n";
    // for (int i = 0; i < net.weights.size(); i++){
    //     for (int j = 0; j < net.weights[i].size(); j++){
    //         for (int k = 0; k < net.weights[i][j].size(); k++){
    //             cout << net.weights[i][j][k] << "\n";
    //         }
    //     }
    // }

    // return 0;

    // float L = 0.0;
    // for (int c = 0; c < 100000; c++){
    //     float loss = 0.0;
    //     for (int i = 0; i <= 1; i++){
    //         for (int j = 0; j <= 1; j++){
    //             net = train(net, {i/1.0f, j/1.0f}, {((int)(i!=j))/1.0f, 1-((int)(i!=j))/1.0f});
    //             loss += net.totalError({((int)(i!=j))/1.0f, 1-((int)(i!=j))/1.0f});
    //         }
    //     }
    //     L = loss;
    // }

    // cout << L << endl;

    // return 0;

    // Image im;
    // im.loadFromFile("Assets/blank.jpg");
    // Texture t;

    // int count = 0;
    
    // while (window.isOpen())
    // {
        
    //     Event event;
    //     while (window.pollEvent(event))
    //     {
    //         if (event.type == Event::Closed){
    //             window.close();
    //         }

    //     }

    //     window.clear();

    //     // for (int i = 0; i <= 1; i++){
    //     //     for (int j = 0; j <= 1; j++){
    //     //         net.input({i/1.0f, j/1.0f});
    //     //         net.backprop({((int)(i!=j))/1.0f, 1-((int)(i!=j))/1.0f});
    //     //     }
    //     // }
    //     // if (count = 999){
    //     //     net.input({1, 1});
    //     //     net.calculate();
    //     //     cout << net.totalError({0, 1}) << "\n";
    //     //     count = 0;
    //     // }
    //     for (int i = 0; i < 800; i++){
    //         for (int j = 0; j < 800; j++){
    //             net.input({i/800.0f, j/800.0f});
    //             net.calculate();
    //             if (net.neurons.back()[0] > net.neurons.back()[1]){
    //                 im.setPixel(i, 800-j, Color::Green);
    //             }
    //             else{
    //                 im.setPixel(i, 800-j, Color::Red);
    //             }
    //         }
    //     }
    //     t.loadFromImage(im);
    //     RectangleShape r(Vector2f(800, 800));
    //     r.setTexture(&t);
    //     window.draw(r);
    //     window.display();
    //     count++;
    // }
    // return 0;
// }