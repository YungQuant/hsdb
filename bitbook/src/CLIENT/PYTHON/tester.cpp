#include "cplot.h"
#include <iostream>
#include <vector>
#include <math.h>

int main()
{
    std::vector<std::vector<double>> X, Y, Z;
    std::vector<double> tx, ty, tz;

    auto fx = [](double i, double j) {
        return -pow(i, 3) - pow(j, 2);
    };

    int m = 10, n = 10;

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            tx.push_back(i);
            ty.push_back(j);
            tz.push_back(fx(i-4,j-4));
        }
        X.push_back(tx);
        Y.push_back(ty);
        Z.push_back(tz);
        std::cout << X.size() << "," << X[0].size() << " - " << Y.size() << "," << Y[0].size() << " - " << Z.size() << "," << Z[0].size() << std::endl;
        tx.clear();
        ty.clear();
        tz.clear();
        cplot::clf();
        cplot::plot_surface(X, Y, Z, "hsv", "");
        cplot::pause(1);
    }


    cplot::show();

    return 0;
}
