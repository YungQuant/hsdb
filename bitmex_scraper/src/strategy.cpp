#include <iostream>
#include "feed.h"
#include "pyhandler.h"

#include <time.h>


void Strategy(feed * self){
    pyhandler::__init__();
    int counter = 0;
    while(true){

        if(self->quant.obook_sync == true){
            self->quant.book("XBTUSD");
            counter += 1;
            std::cout << "Epoch: " << counter << std::endl;
        } else {
            std::cout << "Loading Data" << std::endl;
        }


    }
}


















































 //std::cout << "Strategy Bid Size: " << self->quant.bids["XBTUSD"].size() << "\tAsks Size: " << self->quant.asks["XBTUSD"].size() << std::endl;

            /*
            for(auto kv : self->quant.bids["XBTUSD"]){
                std::string key = kv.first;
                for(auto & t : self->quant.bids["XBTUSD"][key]){
                    std::cout << t << "\t";
                }
                std::cout << std::endl;
            }
            */
