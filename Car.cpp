#include "Car.hpp"

Car::Car(double s, double h, array<double, 2> c){
    size = s;
    heading = h;
    center = {c[0], c[1]};
    corners[0]= {c[0]-s, c[1]-s};
    corners[1] = {c[0]+s, c[1]-s};
    corners[2] = {c[0]+s, c[1]+s};
    corners[3] = {c[0]-s, c[1]+s};
}

void Car::move(double d){
    double dx = d*cos(heading);
    double dy = d*sin(heading);
    center[0] += dx;
    center[1] += dy;
    for (int i = 0; i < corners.size(); i++){
        corners[i][0] += dx;
        corners[i][1] += dy;
    }  
}

void Car::rotate(double a){
    a *= -1;
    for (int i = 0; i < corners.size(); i++){
        double dx = center[0]-corners[i][0];
        double dy = center[1]-corners[i][1];
        double dx_ = dx;
        dx = dx * cos(a) - dy * sin(a);
        dy = dx_ * sin(a) + dy * cos(a);
        corners[i] = {center[0]+dx, center[1]+dy};
    }
    heading += a;
}

vector<double> Car::getDists(Image im){
    array<double, 3> angles = {heading + angleInc, heading, heading - angleInc};
    vector<double> dists = {0, 0, 0};
    for (int n = 0; n < 3; n++){
        double angle = angles[n];
        double dx = 2*cos(angle);
        double dy = 2*sin(angle);

        double x = center[0];
        double y = center[1];
        double dist = 0;
        for (int i = 0; i < 300; i++){
            x += dx;
            y += dy;
            dist += 2.0f;
            int X = int(x);
            int Y = int(y);
            if (X > im.getSize().x || Y > im.getSize().y || X < 0 || Y < 0){
                break;
            }
            if (im.getPixel(X, Y) == Color::Black){
                break;
            }
        }
        dists[n] = dist;
    }
    return dists;
}

array<array<double, 2>, 3> Car::drawRay(Image im){
    array<double, 3> angles = {heading + angleInc, heading, heading - angleInc};
    array<array<double, 2>, 3> hits;
    for (int n = 0; n < 3; n++){
        double angle = angles[n];
        double dx = 2*cos(angle);
        double dy = 2*sin(angle);

        double x = center[0];
        double y = center[1];
        bool hit = false;
        double dist = 0;
        for (int i = 0; i < 300; i++){
            x += dx;
            y += dy;
            dist += 2.0f;
            int X = int(x);
            int Y = int(y);
            if (X > im.getSize().x || Y > im.getSize().y || X < 0 || Y < 0){
                break;
            }
            if (im.getPixel(X, Y) == Color::Black){
                break;
            }
        }
        hits[n] = {x, y};
    }
    return hits;
}


Car::~Car(){

}