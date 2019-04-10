#include <iostream>

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

    std::string heatmap = "viridis";
    int counter = 0;

    while(self->sync == true){
        if(self->quant.sync == true && self->quant.quant_sync == true){

            self->quant.__call__("XBTUSD");

            cplot::clf();
            cplot::plot_surface(self->quant.XY["X"], self->quant.XY["Y"], self->quant.Z, heatmap, "");
            cplot::set_yticklabels(self->quant.lemp);
            cplot::pause(0.00001);
            
        }
    }
    cplot::show();
}
