#include "Car.hpp"

Car::Car(float s, float h){
    size = s;
    heading = h;
    center = {s, s};
    corners[0]= {0, 0};
    corners[1] = {s*2, 0};
    corners[2] = {s*2, s*2};
    corners[3] = {0, s*2};
}

void Car::move(float d){
    float dx = d*cos(heading);
    float dy = d*sin(heading);
    center[0] += dx;
    center[1] += dy;
    for (int i = 0; i < corners.size(); i++){
        corners[i][0] += dx;
        corners[i][1] += dy;
    }  
}

void Car::rotate(float a){
    a *= -1;
    for (int i = 0; i < corners.size(); i++){
        float dx = center[0]-corners[i][0];
        float dy = center[1]-corners[i][1];
        float dx_ = dx;
        dx = dx * cos(a) - dy * sin(a);
        dy = dx_ * sin(a) + dy * cos(a);
        corners[i] = {center[0]+dx, center[1]+dy};
    }
    heading += a;
}

vector<float> Car::getDists(Image im){
    array<float, 3> angles = {heading + angleInc, heading, heading - angleInc};
    vector<float> dists = {0, 0, 0};
    for (int n = 0; n < 3; n++){
        float angle = angles[n];
        double dx = 2*cos(angle);
        double dy = 2*sin(angle);

        double x = center[0];
        double y = center[1];
        bool hit = false;
        float dist = 0;
        for (int i = 0; i < 100; i++){
            x += dx;
            y += dy;
            dist += 2.0f;
            int X = int(x);
            int Y = int(y);
            if (X > im.getSize().x || Y > im.getSize().y || X < 0 || Y < 0){
                return dists;
            }
            if (im.getPixel(X, Y) == Color::Black){
                hit = true;
                break;
            }
        }
        if (!hit) dist = 200;
        dists[n] = dist;
    }
    return dists;
}


Car::~Car(){

}