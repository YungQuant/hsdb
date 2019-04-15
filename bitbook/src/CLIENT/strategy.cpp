#include <iostream>

#include <vector>
#include <string>
#include <time.h>
#include <fstream>

#include "./PYTHON/cplot.h"
#include "./feed.h"

void TestStrategy(feed * self) {
    std::cout << "Position: " << self->trader.auth.nonce() << std::endl;
}

void Strategy(feed * self, std::ofstream & writer){
    std::cout << "Strategy Script Initialized" << std::endl;
    cplot::tight_layout();


    std::vector<std::string> heatmap = {"viridis", "gnuplot", "hsv"};
    std::vector<std::string> colors = {"red","orange","yellow","green","blue","purple"};
    int counter = 0;

    while(self->sync == true){
        for(auto & i : {-20,-10,-5,0,5,10,20,10,5,0,-5,-10}){
            if(self->quant.sync == true && self->quant.quant_sync == true){

                self->quant.__call__("XBTUSD");

                if(self->quant.plot_sync == true){
                    if(counter == heatmap.size()){
                        counter = 0;
                    }
                    cplot::clf();
                    cplot::plot_surface(self->quant.XY["X"], self->quant.XY["Y"], self->quant.Z, heatmap[counter], "");
                    //cplot::plot_scatter(self->quant.XY["X"], self->quant.XY["Y"], self->quant.Z, colors[counter]);

                    cplot::set_yticklabels(self->quant.lemp);
                    cplot::view_init(31, i);
                    cplot::grid(false);
                    cplot::pause(0.0026);
                    counter += 1;
                }
            }
        }
    }
    cplot::show();
}
